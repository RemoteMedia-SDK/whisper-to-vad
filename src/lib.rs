//! WhisperToVadNode as a standalone Path 3 loadable plugin.
//!
//! Runs an ONNX regressor (ridge or MLP) over Whisper encoder hidden
//! states and emits a per-input `{valence, arousal, dominance, intensity}`
//! JSON envelope. Originally lived in `remotemedia-core` under the
//! `affect-listener-face` feature; extracted here so the host crate
//! doesn't drag in `ort` just for this single regressor.
//!
//! ## Node types exported
//!
//!   WhisperToVadNode — Tensor → Json{valence, arousal, dominance, intensity, ...}

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, OnceCell};

use remotemedia_plugin_sdk::abi_stable::sabi_trait::TD_Opaque;
use remotemedia_plugin_sdk::abi_stable::std_types::{ROk, RResult, RString};
use remotemedia_plugin_sdk::adapter::StreamingNodeFfiAdapter;
use remotemedia_plugin_sdk::traits::streaming::AsyncStreamingNode;
use remotemedia_plugin_sdk::types::{Error, RuntimeData};
use remotemedia_plugin_sdk::{FfiNodeBox, FfiNodeFactory, FfiNode_TO};

use ort::{
    execution_providers::CPUExecutionProvider,
    session::{Session, SessionOutputs},
    value::Tensor,
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WhisperToVadConfig {
    #[serde(alias = "modelPath")]
    pub model_path: PathBuf,
    #[serde(alias = "embedDim")]
    pub embed_dim: usize,
    pub pool: PoolMode,
    pub intensity: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PoolMode {
    Mean,
    Last,
}

impl Default for WhisperToVadConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from(
                "tools/affect_calibration/artifacts/whisper_to_vad_ridge.onnx",
            ),
            embed_dim: 1280,
            pool: PoolMode::Mean,
            intensity: 0.4,
        }
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

pub struct WhisperToVadNode {
    config: WhisperToVadConfig,
    session: OnceCell<Arc<Mutex<Session>>>,
}

impl WhisperToVadNode {
    pub fn with_config(config: WhisperToVadConfig) -> Self {
        Self {
            config,
            session: OnceCell::new(),
        }
    }

    async fn get_or_init_session(&self) -> Result<&Arc<Mutex<Session>>, Error> {
        self.session
            .get_or_try_init(|| async {
                let path = &self.config.model_path;
                if !path.exists() {
                    return Err(Error::Execution(format!(
                        "WhisperToVadNode: model not found at {} (override via params.model_path)",
                        path.display()
                    )));
                }
                tracing::info!(model = %path.display(), "Loading whisper→VAD ONNX model");
                let session = Session::builder()
                    .map_err(|e| Error::Execution(format!("ort builder: {e}")))?
                    .with_execution_providers([CPUExecutionProvider::default().build()])
                    .map_err(|e| Error::Execution(format!("ort EP: {e}")))?
                    .commit_from_file(path)
                    .map_err(|e| Error::Execution(format!("ort load: {e}")))?;
                Ok(Arc::new(Mutex::new(session)))
            })
            .await
    }

    async fn run_regression(&self, embed: &[f32]) -> Result<[f32; 3], Error> {
        let session_arc = self.get_or_init_session().await?;
        let mut session = session_arc.lock().await;
        let d = self.config.embed_dim;
        let input = Tensor::from_array(([1usize, d], embed.to_vec()))
            .map_err(|e| Error::Execution(format!("ort input tensor: {e}")))?;
        let outputs: SessionOutputs = session
            .run(ort::inputs!["whisper_embed" => input])
            .map_err(|e| Error::Execution(format!("ort run: {e}")))?;
        let (_, vad) = outputs["vad"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Execution(format!("ort extract vad: {e}")))?;
        if vad.len() < 3 {
            return Err(Error::Execution(format!(
                "WhisperToVadNode: ONNX returned {} values, expected 3",
                vad.len()
            )));
        }
        Ok([vad[0], vad[1], vad[2]])
    }

    fn pool_embedding(&self, samples: &[f32], n_frames: usize, d: usize) -> Vec<f32> {
        match self.config.pool {
            PoolMode::Last => samples[(n_frames - 1) * d..n_frames * d].to_vec(),
            PoolMode::Mean => {
                let mut out = vec![0.0_f32; d];
                for t in 0..n_frames {
                    let row = &samples[t * d..(t + 1) * d];
                    for (acc, &v) in out.iter_mut().zip(row.iter()) {
                        *acc += v;
                    }
                }
                let inv = 1.0_f32 / (n_frames as f32);
                for x in out.iter_mut() {
                    *x *= inv;
                }
                out
            }
        }
    }
}

#[async_trait]
impl AsyncStreamingNode for WhisperToVadNode {
    fn node_type(&self) -> &str {
        "WhisperToVadNode"
    }

    async fn process(&self, _data: RuntimeData) -> Result<RuntimeData, Error> {
        Err(Error::Execution(
            "WhisperToVadNode is streaming-only — use process_streaming()".into(),
        ))
    }

    async fn process_streaming<F>(
        &self,
        data: RuntimeData,
        _session_id: Option<String>,
        mut callback: F,
    ) -> Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> Result<(), Error> + Send,
    {
        let (raw_bytes, shape, dtype, metadata) = match &data {
            RuntimeData::Tensor {
                data,
                shape,
                dtype,
                metadata,
            } => (data.clone(), shape.clone(), *dtype, metadata.clone()),
            _ => {
                callback(data)?;
                return Ok(1);
            }
        };

        if dtype != 0 {
            return Err(Error::InvalidData(format!(
                "WhisperToVadNode: expected f32 tensor (dtype=0), got dtype={dtype}"
            )));
        }
        let total_floats = raw_bytes.len() / 4;
        let samples: Vec<f32> = raw_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if samples.len() != total_floats {
            return Err(Error::InvalidData(
                "WhisperToVadNode: tensor byte length not divisible by 4".into(),
            ));
        }

        let d = self.config.embed_dim;
        let n_frames = match shape.as_slice() {
            [d_only] if (*d_only as usize) == d => 1,
            [n, w] if (*w as usize) == d => *n as usize,
            [b, n, w] if *b == 1 && (*w as usize) == d => *n as usize,
            _ => {
                return Err(Error::InvalidData(format!(
                    "WhisperToVadNode: tensor shape {shape:?} doesn't match [{d}], [T, {d}], or [1, T, {d}]"
                )));
            }
        };
        if n_frames == 0 {
            return Err(Error::InvalidData(
                "WhisperToVadNode: zero-frame tensor".into(),
            ));
        }
        if samples.len() != n_frames * d {
            return Err(Error::InvalidData(format!(
                "WhisperToVadNode: expected {} floats for shape {shape:?}, got {}",
                n_frames * d,
                samples.len()
            )));
        }

        let pooled = if n_frames == 1 {
            samples
        } else {
            self.pool_embedding(&samples, n_frames, d)
        };

        let [v, a, dom] = self.run_regression(&pooled).await?;

        let intensity = metadata
            .as_ref()
            .and_then(|m| m.get("intensity").and_then(|x| x.as_f64()))
            .map(|x| x as f32)
            .unwrap_or(self.config.intensity);

        callback(RuntimeData::Json(serde_json::json!({
            "kind": "vad",
            "valence": v,
            "arousal": a,
            "dominance": dom,
            "intensity": intensity,
            "source": "whisper_to_vad",
            "n_frames": n_frames,
        })))?;
        Ok(1)
    }
}

// ---------------------------------------------------------------------------
// Factory + plugin registration
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct WhisperToVadNodeFactory;

impl FfiNodeFactory for WhisperToVadNodeFactory {
    fn node_type(&self) -> RString {
        RString::from("WhisperToVadNode")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let cfg: WhisperToVadConfig = serde_json::from_str(params.as_str()).unwrap_or_default();
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(WhisperToVadNode::with_config(cfg)),
            TD_Opaque,
        ))
    }
}

remotemedia_plugin_sdk::plugin_export!(WhisperToVadNodeFactory);

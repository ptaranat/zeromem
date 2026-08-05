#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Config {
    /// PPR damping (paper: gamma = 0.6).
    pub gamma: f32,
    /// Primary-view fusion weight (paper: rho = 0.6).
    pub rho: f32,
    /// Evidence budget after calibration (paper: top-5).
    pub top_k: usize,
    /// Candidate pool per view before fusion.
    pub candidate_k: usize,
    /// Entity activation propagation steps.
    pub propagation_steps: usize,
    /// Turns per window.
    pub window_size: usize,
    /// New episode when the inter-window gap exceeds this.
    pub episode_gap_secs: i64,
    /// New episode when adjacent window centroid similarity drops below this.
    pub episode_sim_threshold: f32,
    /// Half-width of a local span around a selected turn.
    pub local_span: usize,
    /// Minimum cosine to align a query entity to a graph entity.
    pub align_threshold: f32,
    /// PPR power iteration limit.
    pub ppr_iters: usize,
    /// Episodes kept in the coarse-to-fine pass.
    pub episode_beam: usize,
    /// Windows kept in the coarse-to-fine pass.
    pub window_beam: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gamma: 0.6,
            rho: 0.6,
            top_k: 5,
            candidate_k: 24,
            propagation_steps: 2,
            window_size: 4,
            episode_gap_secs: 6 * 3600,
            episode_sim_threshold: 0.35,
            local_span: 2,
            align_threshold: 0.55,
            ppr_iters: 30,
            episode_beam: 4,
            window_beam: 8,
        }
    }
}

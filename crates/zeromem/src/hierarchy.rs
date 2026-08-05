//! Temporal hierarchy, paper eq 5: turns -> windows -> episodes. Episodes
//! split on session change, time gap, or centroid similarity drop.

use crate::config::Config;
use crate::embed::{cosine, l2_normalize};

#[derive(Debug, Clone)]
pub struct Window {
    pub episode: u32,
    pub start: u32,
    pub end: u32,
    pub centroid: Vec<f32>,
    n: usize,
}

#[derive(Debug, Clone)]
pub struct Episode {
    pub session_id: String,
    pub start: u32,
    pub end: u32,
}

#[derive(Default)]
pub struct Hierarchy {
    pub windows: Vec<Window>,
    pub episodes: Vec<Episode>,
    pub turn_window: Vec<u32>,
    pub turn_episode: Vec<u32>,
    last_ts: i64,
    last_session: Option<String>,
}

impl Hierarchy {
    pub fn push_turn(&mut self, turn: u32, session_id: &str, ts: i64, emb: &[f32], cfg: &Config) {
        let session_changed = self.last_session.as_deref() != Some(session_id);
        let gap = !session_changed && ts.saturating_sub(self.last_ts) > cfg.episode_gap_secs;

        let window_full = self
            .windows
            .last()
            .map_or(true, |w| (w.end - w.start + 1) as usize >= cfg.window_size);

        if session_changed || gap || window_full {
            let new_episode = session_changed || gap || self.episodes.is_empty() || {
                // Compare the finished window against its episode's previous window.
                let last = self.windows.last().unwrap();
                let prev = self.windows.len().checked_sub(2).map(|i| &self.windows[i]);
                match prev {
                    Some(p) if p.episode == last.episode => {
                        let mut a = last.centroid.clone();
                        let mut b = p.centroid.clone();
                        l2_normalize(&mut a);
                        l2_normalize(&mut b);
                        cosine(&a, &b) < cfg.episode_sim_threshold
                    }
                    _ => false,
                }
            };
            let episode = if new_episode {
                self.episodes.push(Episode {
                    session_id: session_id.to_string(),
                    start: turn,
                    end: turn,
                });
                (self.episodes.len() - 1) as u32
            } else {
                (self.episodes.len() - 1) as u32
            };
            self.windows.push(Window {
                episode,
                start: turn,
                end: turn,
                centroid: emb.to_vec(),
                n: 1,
            });
        } else {
            let w = self.windows.last_mut().unwrap();
            w.end = turn;
            for (c, x) in w.centroid.iter_mut().zip(emb) {
                *c = (*c * w.n as f32 + x) / (w.n as f32 + 1.0);
            }
            w.n += 1;
        }

        let wid = (self.windows.len() - 1) as u32;
        let eid = self.windows[wid as usize].episode;
        self.episodes[eid as usize].end = turn;
        self.turn_window.push(wid);
        self.turn_episode.push(eid);
        self.last_ts = ts;
        self.last_session = Some(session_id.to_string());
    }

    pub fn windows_of_episode(&self, episode: u32) -> impl Iterator<Item = (u32, &Window)> {
        self.windows
            .iter()
            .enumerate()
            .filter(move |(_, w)| w.episode == episode)
            .map(|(i, w)| (i as u32, w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config { window_size: 2, episode_gap_secs: 100, ..Config::default() }
    }

    #[test]
    fn windows_fill_then_roll() {
        let mut h = Hierarchy::default();
        let c = cfg();
        let e = vec![1.0, 0.0];
        for i in 0..4u32 {
            h.push_turn(i, "s1", 10 + i as i64, &e, &c);
        }
        assert_eq!(h.windows.len(), 2);
        assert_eq!(h.episodes.len(), 1);
        assert_eq!(h.turn_window, vec![0, 0, 1, 1]);
    }

    #[test]
    fn session_change_and_gap_split_episodes() {
        let mut h = Hierarchy::default();
        let c = cfg();
        let e = vec![1.0, 0.0];
        h.push_turn(0, "s1", 10, &e, &c);
        h.push_turn(1, "s1", 500, &e, &c); // gap
        h.push_turn(2, "s2", 501, &e, &c); // session change
        assert_eq!(h.episodes.len(), 3);
        assert_eq!(h.episodes[2].session_id, "s2");
    }
}

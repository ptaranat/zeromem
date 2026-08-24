use std::collections::HashSet;
use std::sync::OnceLock;

pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|t| !t.is_empty())
        .map(|t| t.trim_matches('\'').to_lowercase())
        .filter(|t| !t.is_empty())
        .collect()
}

pub fn content_tokens(text: &str) -> Vec<String> {
    tokenize(text)
        .into_iter()
        .filter(|t| !is_stopword(t) && t.len() > 1)
        .collect()
}

/// Naive splitter: terminator followed by whitespace, or newline.
pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '\n'
            || ((c == '.' || c == '!' || c == '?')
                && bytes
                    .get(i + 1)
                    .map_or(true, |b| (*b as char).is_whitespace()))
        {
            let s = text[start..=i].trim();
            if !s.is_empty() {
                out.push(s);
            }
            start = i + 1;
        }
        i += 1;
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

pub fn is_stopword(w: &str) -> bool {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| STOPWORDS.iter().copied().collect())
        .contains(w)
}

static STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "will", "just", "should", "now", "i", "me", "my",
    "myself", "we", "our", "ours", "you", "your", "yours", "he", "him", "his", "she", "her",
    "hers", "it", "its", "they", "them", "their", "theirs", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "would", "could", "ought", "as", "until",
    "because", "s", "t", "don", "let", "us", "also", "get", "got", "like", "yeah", "yes", "okay",
    "ok", "really", "know", "think", "going", "go", "one", "well", "much", "still", "back", "even",
    "want", "said", "say", "told", "tell",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_basic() {
        assert_eq!(
            tokenize("Carrie's dog, Lychee!"),
            vec!["carrie's", "dog", "lychee"]
        );
    }

    #[test]
    fn sentences_split_on_terminators() {
        let s = split_sentences("I moved to Jersey City. It was hot! Really?");
        assert_eq!(s, vec!["I moved to Jersey City.", "It was hot!", "Really?"]);
    }

    #[test]
    fn abbreviation_period_not_split_mid_token() {
        let s = split_sentences("v2.5 shipped today");
        assert_eq!(s, vec!["v2.5 shipped today"]);
    }
}

/// Tiny copy-task generator for sanity-training the transformer.
///
/// Reserves token id 0 as BOS. Returns:
///   - `src`:    the sequence to copy
///   - `tgt_in`: decoder input  = [BOS, src[0..n-1]]
///   - `labels`: decoder target = src
pub struct CopyTask {
    pub vocab: usize,
    pub seq_len: usize,
    state: u64,
}

impl CopyTask {
    pub const BOS: usize = 0;

    pub fn new(vocab: usize, seq_len: usize, seed: u64) -> Self {
        assert!(vocab >= 2, "need at least BOS + one real token");
        Self { vocab, seq_len, state: seed.wrapping_add(0x9E3779B97F4A7C15) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn sample(&mut self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let real_vocab = self.vocab - 1;
        let src: Vec<usize> = (0..self.seq_len)
            .map(|_| 1 + (self.next_u64() as usize) % real_vocab)
            .collect();

        let mut tgt_in = Vec::with_capacity(self.seq_len);
        tgt_in.push(Self::BOS);
        tgt_in.extend(src.iter().take(self.seq_len - 1).copied());

        let labels = src.clone();
        (src, tgt_in, labels)
    }
}

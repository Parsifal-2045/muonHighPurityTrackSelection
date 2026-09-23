// forest_bench.cc - standalone single-thread timing of the CMSSW compact-forest traversal:
// file layout (per track) vs packed depth-first layout (per track and tree-major batches of 8).
// Build: g++ -O2 -std=c++17 -o forest_bench forest_bench.cc
// Run:   ./forest_bench <model_compact.bin> <features.f32 (row-major float32)> <n features>
// Old (file arrays, BFS) vs new (packed DFS) compact-forest traversal, single thread.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
struct Forest {
  int32_t n, t; float base; std::vector<int8_t> feat; std::vector<float> val; std::vector<int32_t> l, r, roots;
  struct P { float val; uint32_t meta; }; std::vector<P> nodes; std::vector<uint32_t> proots;
  uint32_t emit(int32_t k) { uint32_t i = nodes.size(); nodes.push_back({val[k], 0xffu});
    if (feat[k] >= 0) { emit(l[k]); uint32_t rr = emit(r[k]); nodes[i].meta = (rr << 8) | uint32_t(feat[k]); } return i; }
  void load(const char* p) { std::ifstream in(p, std::ios::binary); in.read((char*)&n, 4); in.read((char*)&t, 4); in.read((char*)&base, 4);
    feat.resize(n); val.resize(n); l.resize(n); r.resize(n); roots.resize(t);
    in.read((char*)feat.data(), n); in.read((char*)val.data(), 4LL * n); in.read((char*)l.data(), 4LL * n); in.read((char*)r.data(), 4LL * n); in.read((char*)roots.data(), 4LL * t);
    for (auto rt : roots) proots.push_back(emit(rt)); }
  float evalOld(const float* x) const { float m = base; for (auto rt : roots) { int32_t k = rt; while (feat[k] >= 0) k = (x[feat[k]] < val[k]) ? l[k] : r[k]; m += val[k]; } return 1.f / (1.f + std::exp(-m)); }
  float evalNew(const float* x) const { const P* nd = nodes.data(); float m = base; for (auto rt : proots) { uint32_t i = rt, meta = nd[i].meta;
      while ((meta & 0xffu) != 0xffu) { i = (x[meta & 0xffu] < nd[i].val) ? i + 1 : (meta >> 8); meta = nd[i].meta; } m += nd[i].val; } return 1.f / (1.f + std::exp(-m)); }
  // tree-major over a batch of tracks; per-track accumulation order unchanged
  void evalBatch(const float* X, size_t n, size_t nf, float* out, float* margin) const { const P* nd = nodes.data();
    for (size_t j = 0; j < n; ++j) margin[j] = base;
    for (auto rt : proots) for (size_t j = 0; j < n; ++j) { const float* x = X + j * nf; uint32_t i = rt, meta = nd[i].meta;
        while ((meta & 0xffu) != 0xffu) { i = (x[meta & 0xffu] < nd[i].val) ? i + 1 : (meta >> 8); meta = nd[i].meta; } margin[j] += nd[i].val; }
    for (size_t j = 0; j < n; ++j) out[j] = 1.f / (1.f + std::exp(-margin[j])); }
};
int main(int argc, char** argv) {
  Forest f; f.load(argv[1]); int nf = atoi(argv[3]);
  std::ifstream in(argv[2], std::ios::binary | std::ios::ate); size_t sz = in.tellg(); in.seekg(0);
  std::vector<float> X(sz / 4); in.read((char*)X.data(), sz); size_t ntrk = X.size() / nf;
  size_t diff = 0; for (size_t i = 0; i < ntrk; ++i) diff += f.evalOld(&X[i * nf]) != f.evalNew(&X[i * nf]);
  auto bench = [&](bool nw) { int reps = 0; double sink = 0; auto t0 = std::chrono::steady_clock::now(); double el = 0;
    do { for (size_t i = 0; i < ntrk; ++i) sink += nw ? f.evalNew(&X[i * nf]) : f.evalOld(&X[i * nf]); ++reps;
         el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); } while (el < 2.0);
    return std::make_pair(el / (reps * ntrk) * 1e6, sink); };
  auto o = bench(false); auto n = bench(true);
  size_t bs = 8; std::vector<float> out(bs), mg(bs); size_t bdiff = 0;
  for (size_t i = 0; i + bs <= ntrk; i += bs) { f.evalBatch(&X[i * nf], bs, nf, out.data(), mg.data()); for (size_t j = 0; j < bs; ++j) bdiff += out[j] != f.evalNew(&X[(i + j) * nf]); }
  int reps = 0; double sink = 0; auto t0 = std::chrono::steady_clock::now(); double el = 0;
  do { for (size_t i = 0; i + bs <= ntrk; i += bs) { f.evalBatch(&X[i * nf], bs, nf, out.data(), mg.data()); sink += out[0]; } ++reps;
       el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); } while (el < 2.0);
  double batch_us = el / (reps * (ntrk / bs * bs)) * 1e6;
  printf("   batch of %zu (tree-major): %.1f us/track (%.2fx vs old), mismatches vs per-track %zu\n", bs, batch_us, o.first / batch_us, bdiff);
  printf("%-60s tracks=%zu trees=%d nodes=%d  old %.1f us/track  new %.1f us/track  speedup %.2fx  score mismatches %zu\n", argv[1], ntrk, f.t, f.n, o.first, n.first, o.first / n.first, diff);
}

"""
游戏领域RAG知识库 检索效果评测脚本（知更鸟/GameRobin）
方法论与服装版一致：自建测试集 -> 多配置(切分粒度 x top-k) -> 命中率/MRR -> 未命中归因

使用前提：
  1. 设置环境变量 DASHSCOPE_API_KEY
  2. 评测集文件 game_eval_testset.tsv 与本脚本同目录（格式：编号\t问题\t期望命中标签）
  3. 知识文档位于 ./data/游戏/*.txt

运行：python eval_game.py
输出：控制台打印 3x4=12 组配置矩阵 + 推荐配置的未命中样本归因清单
"""
import os
import time

import config_data as config
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = './data/游戏'
TESTSET = 'game_eval_testset.tsv'
# 3种切分粒度（chunk_size），overlap 固定为其 1/10
CHUNK_SIZES = [300, 500, 1000]
TOP_KS = [1, 2, 3, 5]
# 线上推荐配置（评测后如结论不同请同步修改 config_data.py）
BEST_CHUNK, BEST_K = 500, 3

EMBEDDINGS = DashScopeEmbeddings(model='text-embedding-v4')


def load_docs():
    docs = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(DATA_DIR, fname), 'r', encoding='utf-8') as f:
            docs.append((fname, f.read()))
    return docs


def load_testset():
    items = []
    with open(TESTSET, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('编号'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                items.append({'id': parts[0], 'q': parts[1], 'tag': parts[2]})
    return items


def build_store(chunks, metas):
    """用当前内存数据构建临时 Chroma 检索器"""
    store = Chroma.from_texts(
        chunks,
        embedding=EMBEDDINGS,
        metadatas=metas,
        collection_name=f'eval_{int(time.time()*1000)}',
    )
    return store


def chunk_docs(docs, chunk_size):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=max(30, chunk_size // 10),
        separators=config.separators,
        length_function=len,
    )
    chunks, metas = [], []
    for fname, text in docs:
        for c in splitter.split_text(text):
            chunks.append(c)
            metas.append({'source': fname})
    return chunks, metas


def tag_in_chunk(tag, chunk_text):
    """判断期望标签是否在该块中（容忍切分把标签与主题名拆开：按行匹配）"""
    for line in chunk_text.splitlines():
        if tag in line:
            return True
    return False


def evaluate(docs, testset, chunk_size, k, emb_cache):
    """
    高效评测：语料块只 embedding 一次（按 chunk_size 缓存），
    每个 top-k 仅做 1 次查询批量 embedding + 本地余弦排序，
    12 组配置的 API 调用量 = 3 次批量入库 + 3 次批量查询（而非 12x106 次检索）。
    """
    import numpy as np
    chunks, metas = chunk_docs(docs, chunk_size)
    cache_key = (chunk_size, len(chunks))
    if cache_key not in emb_cache:
        print(f'  [embed] chunk_size={chunk_size}: {len(chunks)} 块入库向量化...')
        vecs = np.array(EMBEDDINGS.embed_documents(chunks), dtype=np.float32)
        emb_cache[cache_key] = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    doc_mat = emb_cache[cache_key]

    q_texts = [it['q'] for it in testset]
    if 'q_mat' not in emb_cache:
        print(f'  [embed] 查询集向量化: {len(q_texts)} 条...')
        q_vecs = np.array(EMBEDDINGS.embed_documents(q_texts), dtype=np.float32)
        emb_cache['q_mat'] = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-9)
    q_mat = emb_cache['q_mat']

    sims = q_mat @ doc_mat.T  # (106, n_chunks) 余弦相似度
    top_idx = np.argsort(-sims, axis=1)[:, :k]

    hit, rr_sum = 0, 0.0
    misses = []
    for row, item in enumerate(testset):
        rank = None
        for pos, ci in enumerate(top_idx[row]):
            if tag_in_chunk(item['tag'], chunks[ci]):
                rank = pos + 1
                break
        if rank:
            hit += 1
            rr_sum += 1.0 / rank
        else:
            misses.append(item)
    n = len(testset)
    return {'hit': hit / n, 'mrr': rr_sum / n, 'misses': misses, 'chunks': len(chunks)}


def main():
    docs = load_docs()
    testset = load_testset()
    print(f'知识库文档: {len(docs)} 篇 | 测试集: {len(testset)} 条\n')
    header = f'{"chunk|k":>10}' + ''.join(f'{("k="+str(k)):>18}' for k in TOP_KS)
    print(header)
    results = {}
    emb_cache = {}
    for cs in CHUNK_SIZES:
        cells = []
        for k in TOP_KS:
            r = evaluate(docs, testset, cs, k, emb_cache)
            results[(cs, k)] = r
            cells.append(f"{r['hit']*100:.1f}%/{r['mrr']:.3f}")
        print(f'{"cs="+str(cs):>10}' + ''.join(f'{c:>18}' for c in cells))
        print()
    # 推荐配置归因
    key = (BEST_CHUNK, BEST_K)
    r = results[key]
    print(f'=== 推荐配置 chunk={BEST_CHUNK}, top-k={BEST_K}: '
          f'命中率 {r["hit"]*100:.1f}% | MRR {r["mrr"]:.3f} | 总块数 {r["chunks"]} ===')
    if r['misses']:
        print('未命中样本（请人工归因：跨文档误召回/表述差异/标签缺失）：')
        for m in r['misses']:
            print(f"  {m['id']}: {m['q']}  (期望标签: {m['tag']})")
    print('\n提示：确认最优配置后，将 config_data.py 的 chunk_size/top_k 同步更新。')


if __name__ == '__main__':
    main()

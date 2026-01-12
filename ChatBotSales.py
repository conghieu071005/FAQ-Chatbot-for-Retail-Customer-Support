
import os, re, sys, math, json, unicodedata, shutil
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings

# ================== CẤU HÌNH ==================
DEFAULT_JSON_PATH = r"C:\VS_code\Data2.json"
PERSIST_DIR = r"C:\Users\nguye\OneDrive\Desktop\Data2"
COLLECTION_NAME = "faq_tfidf_vi"

# Bật/tắt các tuỳ chọn cải thiện
ACCENT_INSENSITIVE = True          # so khớp không dấu cho truy vấn & câu hỏi
USE_STOPWORDS = True               # bỏ từ dừng
ALPHA = 0.7                        # trọng số cosine (0..1). 0.7 = ưu tiên cosine, 0.3 = overlap
TOP_K = 3                          # trả về nội bộ, chọn best trong top-K
MIN_SCORE = 0.12                   # ngưỡng tối thiểu; dưới ngưỡng -> từ chối đoán bừa
BATCH_SIZE = 100                   # <= 166 để tránh lỗi batch của Chroma

# Stopwords tiếng Việt gọn (có thể mở rộng)
VI_STOPWORDS = set("""
là thì mà và với hoặc nhưng của các những cái một một số cho vào ra lên xuống tại từ đến đang sẽ đã được chưa cũng nữa
rằng nếu khi vì bởi nên như vậy v.v v.v. ạ ạ? ạ! à ừ ờ nhé nha nhá hả không ko k hok vậy thôi đi ha
""".split())

# Đồng nghĩa/chuẩn hoá ý (thêm _ để giữ token đơn)
SYNONYMS = {
    "bán": "mặt_hàng",
    "bán_gì": "mặt_hàng",
    "bạn_bán_gì": "mặt_hàng",
    "mặt": "mặt_hàng",
    "mặt_hàng": "mặt_hàng",
    "kinh_doanh": "mặt_hàng",
    "shop_bán_gì": "mặt_hàng",
    "có_bán": "mặt_hàng",
    "có_miễn_phí": "miễn_phí",
    "free": "miễn_phí",
    "freeship": "miễn_phí",
}

PROFANITY = {"đm","địt","cặc","lồn","đụ","đéo","mẹ","bố mày","vkl","vcl"}

# ================== TIỆN ÍCH CHUẨN HOÁ ==================
_word_re = re.compile(r"[a-zA-ZÀ-ỹ0-9_]+", flags=re.UNICODE)

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def normalize_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFC", s).lower()
    return strip_accents(s) if ACCENT_INSENSITIVE else s

def tokenize_raw(s: str) -> List[str]:
    # token đơn giản + nối từ ghép phổ biến
    s = normalize_text(s)
    s = s.replace("bán gì", "bán_gì").replace("mặt hàng", "mặt_hàng").replace("kinh doanh", "kinh_doanh").replace("có miễn phí","có_miễn_phí")
    toks = _word_re.findall(s)
    # map synonyms
    mapped = [SYNONYMS.get(t, t) for t in toks]
    if USE_STOPWORDS:
        mapped = [t for t in mapped if t not in VI_STOPWORDS]
    return mapped

def has_profanity(s: str) -> bool:
    s2 = normalize_text(s)
    return any(bad in s2 for bad in PROFANITY)

# ================== TF-IDF TỰ CÀI ĐẶT ==================
class TfidfVectorizerManual:
    def __init__(self):
        self.vocab: Dict[str,int] = {}
        self.idf: List[float] = []
        self.fitted = False
        self.doc_tokens: List[List[str]] = []  # giữ lại cho overlap

    def fit(self, docs: List[str]):
        docs_tokens = [tokenize_raw(d) for d in docs]
        self.doc_tokens = docs_tokens
        vocab = {}
        for toks in docs_tokens:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab
        N = len(docs_tokens)
        df = [0]*len(vocab)
        for toks in docs_tokens:
            seen = set()
            for t in toks:
                idx = vocab.get(t)
                if idx is not None and idx not in seen:
                    df[idx]+=1
                    seen.add(idx)
        # Smooth IDF
        self.idf = [math.log((1+N)/(1+dfi))+1.0 for dfi in df]
        self.fitted = True

    def transform_one(self, doc: str) -> List[float]:
        assert self.fitted
        toks = tokenize_raw(doc)
        if not toks: return [0.0]*len(self.vocab)
        counts: Dict[int,int] = {}
        for t in toks:
            idx = self.vocab.get(t)
            if idx is not None:
                counts[idx] = counts.get(idx,0)+1
        tfidf = [0.0]*len(self.vocab)
        L = float(len(toks))
        for idx,c in counts.items():
            tfidf[idx] = (c/L)*self.idf[idx]
        return tfidf

    def transform(self, docs: List[str]) -> List[List[float]]:
        return [self.transform_one(d) for d in docs]

# ================== SIMILARITY ==================
def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = na = nb = 0.0
    for x,y in zip(a,b):
        dot += x*y; na += x*x; nb += y*y
    if na==0 or nb==0: return 0.0
    return dot/(math.sqrt(na)*math.sqrt(nb))

def token_overlap_score(q_tokens: List[str], d_tokens: List[str]) -> float:
    if not q_tokens or not d_tokens: return 0.0
    qs, ds = set(q_tokens), set(d_tokens)
    inter = len(qs & ds)
    denom = min(len(qs), len(ds))
    return inter/denom if denom>0 else 0.0

# ================== CHROMADB QUẢN LÝ ==================
def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(allow_reset=False))
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        col = client.create_collection(COLLECTION_NAME)
    return col

def reset_chroma_if_corrupted():
    try:
        _ = get_collection()
    except Exception:
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        os.makedirs(PERSIST_DIR, exist_ok=True)
        _ = get_collection()

def load_faq_json(path: str) -> List[Dict[str,str]]:
    with open(path,"r",encoding="utf-8") as f:
        data = json.load(f)
    out=[]
    for i,item in enumerate(data):
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            out.append({"id": f"faq_{i}", "question": q, "answer": a})
    if not out:
        raise ValueError("JSON không có mục question/answer hợp lệ.")
    return out

def build_or_refresh_index(faq_items: List[Dict[str,str]], vectorizer: TfidfVectorizerManual|None=None) -> TfidfVectorizerManual:
    reset_chroma_if_corrupted()
    col = get_collection()

    questions = [x["question"] for x in faq_items]
    if vectorizer is None:
        vectorizer = TfidfVectorizerManual()
        vectorizer.fit(questions)
    vectors = vectorizer.transform(questions)

    # Xoá dữ liệu cũ (nếu có)
    try:
        existing = col.get(include=["embeddings","metadatas","documents"])
        if existing.get("ids"):
            col.delete(ids=existing["ids"])
    except Exception:
        pass

    # Thêm theo batch
    ids = [x["id"] for x in faq_items]
    metadatas = [{"answer": x["answer"], "question": x["question"]} for x in faq_items]
    for i in range(0, len(faq_items), BATCH_SIZE):
        end = i + BATCH_SIZE
        col.add(
            ids=ids[i:end],
            documents=questions[i:end],
            metadatas=metadatas[i:end],
            embeddings=vectors[i:end],
        )
    print(f"✅ Đã nạp {len(faq_items)} câu hỏi vào ChromaDB (chia {math.ceil(len(faq_items)/BATCH_SIZE)} batch).")
    return vectorizer

def ensure_vectorizer(faq_items: List[Dict[str,str]]):
    return build_or_refresh_index(faq_items)

# ================== TRUY VẤN ==================
def hybrid_score(query: str, doc_index: int, cosine_val: float, vectorizer: TfidfVectorizerManual) -> float:
    # Overlap giữa query và câu hỏi gốc tương ứng
    q_tokens = tokenize_raw(query)
    d_tokens = vectorizer.doc_tokens[doc_index] if 0 <= doc_index < len(vectorizer.doc_tokens) else []
    ov = token_overlap_score(q_tokens, d_tokens)
    return ALPHA * cosine_val + (1.0 - ALPHA) * ov

def answer_question(user_q: str, vectorizer: TfidfVectorizerManual) -> Tuple[str, float, Dict]:
    if has_profanity(user_q):
        return "Mình không thể hỗ trợ với nội dung không phù hợp. Bạn vui lòng hỏi lại theo cách khác nhé.", 0.0, {}

    col = get_collection()
    store = col.get(include=["embeddings","metadatas","documents"])
    if not store["ids"]:
        return "Chưa có dữ liệu FAQ.", 0.0, {}

    q_vec = vectorizer.transform_one(user_q)

    # Tính cosine cho tất cả
    scores = []
    for i, emb in enumerate(store["embeddings"]):
        cos = cosine_sim(q_vec, emb)
        scores.append((i, cos))

    # Lấy top-K theo cosine rồi tính hybrid
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:TOP_K]
    best_idx, best_hscore, best_cos = -1, -1.0, 0.0
    for i, cos in top:
        h = hybrid_score(user_q, i, cos, vectorizer)
        if h > best_hscore:
            best_hscore, best_idx, best_cos = h, i, cos

    if best_hscore < MIN_SCORE:
        return "Mình chưa chắc bạn cần gì. Bạn mô tả rõ hơn (vd: 'có miễn phí vận chuyển nội thành không?') nhé.", best_hscore, {}

    meta = {
        "matched_question": store["documents"][best_idx],
        "answer": store["metadatas"][best_idx]["answer"],
        "cosine": round(float(best_cos), 6),
        "score": round(float(best_hscore), 6),
    }
    return meta["answer"], best_hscore, meta

# ================== MAIN ==================
def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON_PATH
    if not os.path.exists(json_path):
        print(f"❌ Không tìm thấy file JSON: {json_path}")
        return
    faq_items = load_faq_json(json_path)
    vectorizer = ensure_vectorizer(faq_items)

    print("=== ElectroStore FAQ Chatbot (TF-IDF + ChromaDB + Hybrid) ===")
    print("Gõ câu hỏi (tiếng Việt). Gõ 'exit' để thoát.\n")

    while True:
        try:
            q = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break
        if q.lower() in {"exit","quit"}:
            print("Tạm biệt!")
            break
        if not q:
            continue

        ans, score, meta = answer_question(q, vectorizer)
        print(f"Bot: {ans}")
        if meta:
            print(f"(Khớp: \"{meta.get('matched_question','')}\" | cosine={meta.get('cosine',0.0)} | score={meta.get('score',0.0)})\n")
        else:
            print(f"(score={round(score,6)})\n")

# ================== GIAO DIỆN STREAMLIT ==================
import streamlit as st
import os

# Giả sử bạn có các hàm sau (tùy bạn định nghĩa ở nơi khác)
# from your_module import load_faq_json, ensure_vectorizer, answer_question, DEFAULT_JSON_PATH

def run_streamlit_ui():
    st.set_page_config(
        page_title="ElectroStore Chatbot",
        page_icon="⚡",
        layout="centered"
    )

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("## ⚡ **CellPhoneS**")
        st.markdown("""
        **Địa chỉ:** 71 Ngũ Hành Sơn, P. Mỹ An, Q. Ngũ Hành Sơn, Đà Nẵng  
        **Điện thoại:** 0868357896    
        **Email:** Nguyenconghieu071005@gmail.com  
        **Giờ làm việc:** 8:00 - 21:00 (T2 - CN)  
        """)
        st.markdown("💬 *Chatbot hỗ trợ tư vấn sản phẩm và chính sách cửa hàng*")

    # ========== MAIN CONTENT ==========
    st.title("Chào bạn đến với **CellPhoneS**!")
    st.caption("Hỏi về sản phẩm Apple, chính sách bảo hành, trả góp...")

    # Load dữ liệu & vectorizer
    if "vectorizer" not in st.session_state:
        if not os.path.exists(DEFAULT_JSON_PATH):
            st.error(f"Không tìm thấy file dữ liệu: {DEFAULT_JSON_PATH}")
            return

        faq_items = load_faq_json(DEFAULT_JSON_PATH)
        vectorizer = ensure_vectorizer(faq_items)
        st.session_state.vectorizer = vectorizer
        st.session_state.faq_items = faq_items
        st.session_state.history = []

    vectorizer = st.session_state.vectorizer

    # Hiển thị lịch sử chat
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(item["q"])
        with st.chat_message("assistant"):
            st.markdown(item["a"])


    # Ô nhập tin nhắn
    user_q = st.chat_input("Nhập câu hỏi của bạn (gõ 'exit' để thoát)...")

    if user_q:
        if user_q.lower().strip() in {"exit", "quit"}:
            st.stop()

        with st.chat_message("user"):
            st.markdown(user_q)

        ans, score, meta = answer_question(user_q, vectorizer)

        with st.chat_message("assistant"):
            st.markdown(ans)

        # Lưu lịch sử chat
        st.session_state.history.append({"q": user_q, "a": ans, "meta": meta})

if __name__ == "__main__":
    run_streamlit_ui()

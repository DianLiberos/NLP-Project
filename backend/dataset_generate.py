import json
import pickle
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# Cố định seed để tái lập kết quả 100% cho báo cáo khoa học
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INTENT_TO_ID = {"GREETING": 0, "TRADE": 1, "INFO": 2, "HOSTILE": 3, "UNKNOWN": 4}

# 1. NGÂN HÀNG TỪ VỰNG RPG TIẾNG VIỆT
VOCAB = {
    "player_pronoun": [
        "tôi", "tui", "mình", "tao", "em", "anh", "tại hạ", "tiểu đệ", "bổn tọa",
        "lão tử", "bố mày", "anh đây", "anh mày", "người chơi này", "kẻ lữ hành này", "thằng này"
    ],
    "npc_title": [
        "ông chủ", "chủ tiệm", "shop", "lão bản", "tiền bối", "đại sư", "thợ rèn",
        "thương nhân", "bác lái buôn", "chủ quán", "NPC", "lão già", "huynh đài",
        "các hạ", "chưởng quầy", "bà chủ", "anh chủ shop", "thầy pháp", "trưởng làng"
    ],
    "greet_word": [
        "xin chào", "chào", "hello", "hi", "hê lô", "alo", "kính chào", "bái kiến",
        "chào buổi sáng", "chào buổi tối", "yo", "ê", "hú", "chào mừng", "tham kiến"
    ],
    "polite_particle": [
        "nhé", "nha", "ạ", "với", "đi", "nào", "nhỉ", "hả", "đây", "ơi", "nghen", "hen"
    ],
    "item_weapon": [
        "kiếm sắt", "đại đao", "cung gỗ", "trượng phép", "rìu chiến", "nỏ thần",
        "dao găm", "thương bạc", "kiếm ánh sáng", "gậy triệu hồi", "búa tạ",
        "song kiếm", "quạt ba tiêu", "phi tiêu", "súng hỏa mai", "kiếm gãy",
        "đao đồ long", "cung băng giá", "trượng bóng tối", "chùy gai"
    ],
    "item_armor": [
        "khiên gỗ", "giáp sắt", "mũ đồng", "giày tốc độ", "áo choàng tàng hình",
        "găng tay da", "giáp vảy rồng", "khiên hiệp sĩ", "đai lưng titan",
        "áo vải thô", "mũ pháp sư", "giáp gai", "nhẫn kháng phép", "dây chuyền máu"
    ],
    "item_consumable": [
        "bình máu", "bình mana", "thuốc giải độc", "bùa hồi sinh", "đá cường hóa",
        "ngọc rồng", "lông vũ phượng hoàng", "bình thể lực", "cuộn dịch chuyển",
        "thuốc tăng lực", "thịt nướng", "bánh mì", "rượu gạo", "đá khảm",
        "bùa hộ mệnh", "ngọc thuộc tính", "sách kỹ năng", "rương báu vật", "chìa khóa vàng"
    ],
    "item_material": [
        "quặng sắt", "gỗ sồi", "da sói", "nanh độc", "vảy rồng", "lông chim",
        "đá ma thuật", "bụi tiên", "xương khô", "ngọc bích", "thép tinh luyện", "tơ nhện"
    ],
    "currency": [
        "vàng", "bạc", "xu", "kim cương", "ngọc", "đồng", "gold", "zen", "coin", "linh thạch"
    ],
    "trade_verb_buy": [
        "mua", "lấy", "tậu", "nhập", "đổi lấy", "thuê", "chốt", "xúc", "hốt", "sắm"
    ],
    "trade_verb_sell": [
        "bán", "thanh lý", "nhượng lại", "pass lại", "xả", "thu mua", "đẩy đi", "gả"
    ],
    "trade_verb_service": [
        "rèn", "sửa", "nâng cấp", "cường hóa", "ép", "khảm ngọc", "tẩy luyện", "giám định"
    ],
    "location": [
        "rừng phía bắc", "hang nhện", "đầm lầy tử thần", "đỉnh núi tuyết", "làng tân thủ",
        "thành chính", "hầm ngục tối", "thung lũng gió", "đảo hải tặc", "tháp ma thuật",
        "mỏ khoáng sản", "cổng thành phía nam", "đền cổ", "sa mạc lửa", "bến cảng",
        "doanh trại orc", "khu chợ đen", "lâu đài máu", "vực thẳm", "rừng tinh linh"
    ],
    "monster": [
        "goblin", "slime", "sói xám", "nhện độc", "rồng lửa", "quỷ vương", "xương khô",
        "golem đá", "tinh linh hắc ám", "vua hải tặc", "orc chiến binh", "rắn khổng lồ",
        "dơi hút máu", "boss thế giới", "phù thủy bóng đêm", "kỵ sĩ không đầu"
    ],
    "quest_noun": [
        "nhiệm vụ", "quest", "ủy thác", "thử thách", "cốt truyện", "bản đồ kho báu",
        "mật thư", "chứng chỉ thợ săn", "lệnh truy nã", "nhiệm vụ hàng ngày"
    ],
    "insult_noun": [
        "thằng chó", "đồ phế vật", "con lợn", "thằng ngu", "đồ rác rưởi", "lão già dịch",
        "thằng lừa đảo", "đồ ăn hại", "thằng bất tài", "cục nợ", "đồ hút máu",
        "thằng cặn bã", "con gà", "đồ thần kinh", "lũ sâu bọ"
    ],
    "hostile_action": [
        "chém chết", "đập nát tiệm", "đốt quán", "giết", "cướp sạch", "vả vỡ mồm",
        "đấm cho bầm dập", "san phẳng chỗ này", "lột sạch đồ", "băm vằm", "tiễn lên đường"
    ],
    "curse_prefix": [
        "đm", "dm", "đkm", "vcl", "vl", "đéo mẹ", "mẹ kiếp", "bố tổ", "cái đệt", "má nó"
    ],
    "ood_topic": [
        "bitcoin", "ethereum", "cổ phiếu chứng khoán", "giá vàng SJC", "giá xăng hôm nay",
        "trà sữa trân châu", "cơm tấm sườn bì chả", "phở bò", "lẩu thái", "bún đậu mắm tôm",
        "iPhone 16 Pro Max", "card RTX 4090", "xe SH", "MacBook Air", "sổ đỏ nhà đất",
        "Ngoại hạng Anh", "Real Madrid", "Manchester United", "World Cup", "Sơn Tùng MTP"
    ],
    "ood_academic": [
        "code Python", "thuật toán SVM", "mạng nơ ron PyTorch", "đạo hàm tích phân",
        "nhập môn lập trình", "cấu trúc dữ liệu", "lịch sử đảng", "triết học Mác",
        "tiếng Anh IELTS", "vật lý đại cương", "hóa hữu cơ", "giải phương trình bậc hai"
    ],
    "ood_casual": [
        "thời tiết ngoài trời", "mấy giờ rồi", "tổng thống Mỹ là ai", "cách tán gái",
        "mật khẩu wifi", "xổ số kiến thiết chiều nay", "phim Netflix hay",
        "đau bụng quá", "buồn ngủ vãi", "hết tiền đóng học phí", "người yêu cũ"
    ]
}

# Gộp toàn bộ item vào 1 danh sách chung để tiện lấy ngẫu nhiên
VOCAB["any_item"] = (
    VOCAB["item_weapon"] + VOCAB["item_armor"] + VOCAB["item_consumable"] + VOCAB["item_material"]
)

# 2. KHUNG NGỮ PHÁP ĐA NHÂN CÁCH + HARD NEGATIVES
# Khung câu dành cho tập TRAIN
TRAIN_TEMPLATES = {
    "GREETING": [
        "{greet_word} {npc_title}, chúc một ngày tốt lành!",
        "{greet_word} {npc_title}, dạo này buôn bán thế nào rồi?",
        "{npc_title} ơi, {player_pronoun} mới tới vùng này, {greet_word} {polite_particle}!",
        "{greet_word}, có ai ở trong tiệm không vậy?",
        "{player_pronoun} ghé thăm {npc_title} chút nè, khỏe không?",
        "Lâu quá không gặp {npc_title}, vẫn khỏe chứ {polite_particle}?",
        "{greet_word} buổi sáng {npc_title}, mở hàng sớm thế!",
        "Hú {npc_title}, {player_pronoun} quay lại rồi đây!",
        "Cho {player_pronoun} gửi lời {greet_word} tới {npc_title} {polite_particle}.",
        "{greet_word} đại ca, nhìn {npc_title} hôm nay phong độ đấy!",
        # Hard negative (Có nhắc tới item/địa danh nhưng chỉ là câu chào xã giao)
        "{greet_word} {npc_title}, {player_pronoun} vừa đi săn ở {location} về ghé chào ông một tiếng!",
        "{greet_word} thợ rèn, tiệm đầy {item_weapon} nhìn uy tín quá ha!"
    ],
    "TRADE": [
        "{npc_title} ơi, bán cho {player_pronoun} một cái {any_item} với.",
        "Cây {item_weapon} này giá bao nhiêu {currency} vậy {npc_title}?",
        "{player_pronoun} muốn {trade_verb_buy} ít {item_consumable}, tính tổng hết nhiêu {currency}?",
        "Ở đây có thu mua {item_material} không, {player_pronoun} muốn {trade_verb_sell} một ít.",
        "Giúp {player_pronoun} {trade_verb_service} cái {item_armor} này lên cấp với {npc_title}.",
        "Báo giá cho {player_pronoun} bộ {item_armor} với cây {item_weapon} đi {npc_title}.",
        "{player_pronoun} hết sạch {item_consumable} rồi, lấy gấp cho 10 bình tính bằng {currency} {polite_particle}.",
        "Đổi 50 {item_material} lấy cái {item_weapon} kia được không {npc_title}?",
        "Mở kho hàng ra cho {player_pronoun} xem có {any_item} nào xịn không nào.",
        "Giảm giá cái {any_item} này chút đi {npc_title}, {player_pronoun} {trade_verb_buy} liền!",
        # Hard negative (Giọng cộc lốc nhưng bản chất là mua bán thuần túy)
        "Lấy nhanh cái {any_item} coi, {player_pronoun} trả đủ {currency} chứ có xin xỏ đâu!",
        "Đắt cũng được, {trade_verb_buy} luôn cây {item_weapon} này, gói lại nhanh lên!"
    ],
    "INFO": [
        "{npc_title} cho {player_pronoun} hỏi đường đi tới {location} ở hướng nào vậy?",
        "Làm sao để tiêu diệt được con {monster} ở {location} thế {npc_title}?",
        "{npc_title} có {quest_noun} nào mới cho {player_pronoun} nhận không?",
        "Nghe đồn vùng {location} có {any_item} quý hiếm, tin đó có thật không {npc_title}?",
        "Muốn tìm gặp {npc_title} khác để học nghề thì phải đi đâu hả ông?",
        "Con {monster} thường xuất hiện vào thời gian nào trong ngày vậy {npc_title}?",
        "Cấp độ bao nhiêu mới đủ sức vào {location} làm {quest_noun} vậy {polite_particle}?",
        "{player_pronoun} bị kẹt ở {quest_noun} này rồi, {npc_title} gợi ý cách giải với!",
        "Nguyên liệu {item_material} thì đánh con {monster} nào mới rớt ra vậy {npc_title}?",
        "Kể cho {player_pronoun} nghe truyền thuyết về {location} đi {npc_title}.",
        # Hard negative (Nhắc tới tiền/vật phẩm nhưng là hỏi thông tin/cơ chế game)
        "Đánh con {monster} ở {location} thì rớt khoảng bao nhiêu {currency} và {item_material} vậy?",
        "Lịch sử của thanh {item_weapon} huyền thoại này bắt nguồn từ đâu hả {npc_title}?"
    ],
    "HOSTILE": [
        "{curse_prefix} {npc_title}, mày nhìn cái gì mà nhìn, muốn ăn đòn à?",
        "Đồ {insult_noun}, hôm nay {player_pronoun} sẽ {hostile_action} của mày!",
        "Cút ngay khỏi mắt {player_pronoun} trước khi tao nổi điên lên đồ {insult_noun}!",
        "Mày là cái thá gì mà dám lên mặt dạy đời {player_pronoun} hả {insult_noun}?",
        "{curse_prefix}, tin tao gọi anh em tới {hostile_action} luôn không hả {npc_title}?",
        "Nhìn mặt thằng {npc_title} này ngứa mắt thật sự, bước ra đây solo!",
        "Thứ {insult_noun} vô dụng, sống chật đất của làng!",
        "Đưa hết tài sản đây không tao {hostile_action} bây giờ đồ {insult_noun}!",
        # Hard negatives (Chứa từ TRADE/GREETING/INFO nhưng thực chất là chửi bới/đe dọa)
        "{curse_prefix} {npc_title} bán {any_item} giá cắt cổ thế à, định cướp cạn hả {insult_noun}?",
        "{greet_word} cái con khỉ khô, đồ {insult_noun} lừa đảo bán {item_weapon} dỏm cho bố!",
        "Hỏi đường đi {location} có một câu mà thái độ lồi lõm à thằng {insult_noun} kia?",
        "Thứ {any_item} rác rưởi này mà đòi đổi {currency} á, tao đập vào mặt mày giờ!"
    ],
    "UNKNOWN": [
        "{npc_title} ơi, hôm nay giá {ood_topic} ngoài đời là bao nhiêu tiền vậy?",
        "Giải giúp {player_pronoun} bài tập môn {ood_academic} này với đau đầu quá!",
        "Tối nay trận bóng {ood_topic} đá lúc mấy giờ vậy {npc_title}?",
        "{player_pronoun} đang {ood_casual}, chán chẳng buồn chơi game nữa.",
        "Ở quanh đây có quán nào bán {ood_topic} ship tận nhà không nhỉ?",
        "Làm sao để fix bug trong {ood_academic} vậy, chạy toàn báo lỗi thôi!",
        "Cho hỏi {ood_casual} thế nào rồi, mai có mưa không?",
        "asdfghjkl 123456789 alo alo một hai ba bốn test mic!",
        # Hard negatives (Dùng cấu trúc mua bán/hỏi han của game nhưng thực thể ngoài đời)
        "{npc_title} có {trade_verb_buy} hay {trade_verb_sell} {ood_topic} trả góp không?",
        "Làm {quest_noun} gì để qua môn {ood_academic} đạt điểm A bây giờ hả {npc_title}?",
        "Đổi 1000 {currency} trong game lấy {ood_topic} ngoài đời thật được không shop?",
        "Chỉ đường cho {player_pronoun} đi mua {ood_topic} về ăn chống đói cái coi!"
    ]
}

# Khung câu ĐỘC QUYỀN chỉ xuất hiện trong tập TEST
TEST_TEMPLATES = {
    "GREETING": [
        "Ồ, {npc_title} vẫn đứng canh tiệm đấy à, {greet_word} nhé!",
        "Một ngày mới bình an nha {npc_title}, {player_pronoun} ghé qua điểm danh cái!",
        "Xin phép làm phiền {npc_title} một lát, {greet_word} ngài!",
        "Này {npc_title}, nhận ra khách quen là {player_pronoun} không, {greet_word}!",
        "{greet_word} nha, nhìn sạp {any_item} của {npc_title} hôm nay đông khách gớm!"
    ],
    "TRADE": [
        "Thanh toán cho {player_pronoun} cặp {item_weapon} với 5 lọ {item_consumable} bằng {currency} lẹ đi!",
        "Cần thanh lý gấp đống {item_material} vừa nhặt được, {npc_title} trả được mấy đồng {currency}?",
        "Chi phí để {trade_verb_service} món {item_armor} này tốn hết bao nhiêu {currency} vậy shop?",
        "Có nhận trao đổi {any_item} cũ lấy hàng mới bù thêm {currency} không {npc_title}?",
        "Gói cho {player_pronoun} mấy món {item_consumable} mang đi đường, tính tiền luôn nhé!"
    ],
    "INFO": [
        "Bí kíp nào để vượt qua ải {location} mà không bị {monster} đánh bại vậy {npc_title}?",
        "Muốn hoàn thành {quest_noun} này thì {player_pronoun} cần thu thập đủ mấy cái {item_material}?",
        "Từ đây cuốc bộ tới {location} mất bao lâu hả {npc_title}?",
        "Điểm yếu chí mạng của loài {monster} nằm ở đâu, xin {npc_title} chỉ giáo!",
        "Ai là người giao {quest_noun} khám phá vùng {location} vậy {npc_title}?"
    ],
    "HOSTILE": [
        "Cái tiệm rách nát bán toàn {any_item} phế phẩm, dẹp tiệm đi đồ {insult_noun}!",
        "{curse_prefix}, cút ra chỗ khác trước khi {player_pronoun} {hostile_action}!",
        "Đừng để tao nóng máu, cái mạng già của {npc_title} không chịu nổi một đao đâu đồ {insult_noun}!",
        "Bán buôn kiểu lừa lọc thế này thì sớm muộn tao cũng tới {hostile_action} thôi con chó!",
        "Mở mồm nói thêm câu nữa là tao {hostile_action} ngay tại chỗ đấy {insult_noun}!"
    ],
    "UNKNOWN": [
        "Tư vấn giùm {player_pronoun} xem nên đầu tư vào {ood_topic} hay gửi tiết kiệm ngân hàng?",
        "Thầy giáo bắt nộp đồ án {ood_academic} mà giờ {player_pronoun} chưa viết được chữ nào!",
        "Ngoài đời đang {ood_casual}, tự nhiên chui vô game hỏi {npc_title} làm chi không biết!",
        "Shop có nhận ship {ood_topic} qua ứng dụng giao đồ ăn không thế?",
        "Cho xin cái đánh giá về {ood_topic} năm nay xem có đáng tiền mua không bạn ơi!"
    ]
}

# 3. BỘ BIẾN ĐỔI NHIỄU THỰC TẾ 
TEENCODE_MAP = {
    r"\bkhông\b": ["ko", "k", "khum", "hong", "đéo", "chả"],
    r"\bđược\b": ["dc", "đc", "dk"],
    r"\bbao nhiêu\b": ["bn", "nhiu", "ao nhiêu", "bnhiu", "nhiêu"],
    r"\bông chủ\b": ["og chủ", "ong chu", "shop", "chủ shop"],
    r"\btôi\b": ["tui", "t", "mik", "mình"],
    r"\bmày\b": ["m", "mi", "cậu"],
    r"\bvậy\b": ["z", "v", "zậy", "dậy"],
    r"\bgì\b": ["j", "gì z", "chi"],
    r"\brồi\b": ["r", "rùi", "oy"],
    r"\bnhiệm vụ\b": ["nv", "quest", "q"],
    r"\bvới\b": ["vs", "zới"],
    r"\btiền\b": ["tìn", "xèng", "lúa"],
    r"\bthế nào\b": ["ntn", "sao", "ra sao"],
    r"\bở đâu\b": ["đâu", "chỗ nào", "khúc nào"],
}

FILLER_PREFIXES = ["", "ê ", "nè ", "alo ", "này ", "ủa ", "ơ kìa ", "hú "]
FILLER_SUFFIXES = ["", " nhỉ", " thế", " vcl", " vậy", " đi", " coi", " hả", " nha", " nhé"]


def remove_vn_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt (mô phỏng game thủ gõ phím tắt Unikey)"""
    text = re.sub(r"[đ]", "d", text)
    text = re.sub(r"[Đ]", "D", text)
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def apply_teencode(text: str) -> str:
    """Thay thế từ chuẩn bằng teencode/viết tắt của game thủ"""
    out = text.lower()
    for pattern, replacements in TEENCODE_MAP.items():
        if re.search(pattern, out) and random.random() < 0.75:
            out = re.sub(pattern, random.choice(replacements), out)
    # Bỏ bớt dấu câu ở cuối
    out = re.sub(r"[!?.]+$", "", out)
    return out


def apply_typo_and_slang(text: str) -> str:
    """Tạo lỗi gõ phím ngẫu nhiên + rụng dấu nửa vời + thêm từ đệm"""
    out = apply_teencode(text)
    words = out.split()
    # 50% cơ hội xóa dấu tiếng Việt ở một nửa số từ trong câu
    if len(words) > 3 and random.random() < 0.6:
        for idx in range(len(words)):
            if random.random() < 0.45:
                words[idx] = remove_vn_diacritics(words[idx])
        out = " ".join(words)

    # Thêm từ đệm đầu/cuối
    out = random.choice(FILLER_PREFIXES) + out + random.choice(FILLER_SUFFIXES)
    return re.sub(r"\s+", " ", out).strip()


def generate_4_variants(seed_text: str) -> list[str]:
    """Từ 1 câu gốc (Seed) sinh ra đúng 4 biến thể không trùng lặp"""
    v1_clean = seed_text.strip()
    v2_teencode = apply_teencode(seed_text)
    v3_no_accent = remove_vn_diacritics(v2_teencode if random.random() < 0.5 else v1_clean.lower())
    v4_noisy = apply_typo_and_slang(seed_text)

    variants = [v1_clean, v2_teencode, v3_no_accent, v4_noisy]
    # Đảm bảo nếu vô tình trùng thì thêm biến thể phụ để đủ 4 câu khác nhau
    unique_vars = []
    for idx, v in enumerate(variants):
        if v in unique_vars:
            v = f"{v} {random.choice(['nha', 'nhé', 'đi', 'coi', 'nè', 'hả'])}"
        unique_vars.append(v)
    return unique_vars


# 4. SINH SEED DATA VÀ CHIA TÁCH TRƯỚC KHI AUGMENT
def fill_template(template: str) -> str:
    """Điền ngẫu nhiên các thực thể từ VOCAB vào khung câu"""
    keys = re.findall(r"\{(\w+)\}", template)
    filled = template
    for k in keys:
        if k in VOCAB:
            filled = filled.replace(f"{{{k}}}", random.choice(VOCAB[k]), 1)
    return filled


def build_seed_pool(templates: list[str], target_count: int, forbidden_seeds: set) -> list[str]:
    """Sinh đủ số lượng câu gốc (Seeds) duy nhất, cấm trùng với tập forbidden_seeds"""
    seeds = []
    seen = set(forbidden_seeds)
    attempts = 0
    while len(seeds) < target_count and attempts < target_count * 50:
        attempts += 1
        tpl = random.choice(templates)
        candidate = fill_template(tpl)
        norm_key = remove_vn_diacritics(candidate.lower())
        if norm_key not in seen:
            seen.add(norm_key)
            seeds.append(candidate)
    return seeds


def main():
    print("🚀 Bắt đầu khởi tạo bộ dữ liệu 10.000 mẫu...")

    train_dataset = []
    test_dataset = []
    global_seen_seeds = set()

    # Mục tiêu mỗi Intent:
    # - Train: 400 seeds gốc x 4 biến thể = 1.600 mẫu (x 5 intents = 8.000 mẫu)
    # - Test:  100 seeds gốc x 4 biến thể = 400 mẫu   (x 5 intents = 2.000 mẫu)
    TRAIN_SEEDS_PER_INTENT = 400
    TEST_SEEDS_PER_INTENT = 100

    for intent in INTENT_TO_ID.keys():
        # 1. Sinh 400 câu gốc cho tập TRAIN từ TRAIN_TEMPLATES
        train_seeds = build_seed_pool(
            TRAIN_TEMPLATES[intent], TRAIN_SEEDS_PER_INTENT, global_seen_seeds
        )
        for s in train_seeds:
            global_seen_seeds.add(remove_vn_diacritics(s.lower()))

        # 2. Sinh 100 câu gốc cho tập TEST:
        #    70% lấy từ TEST_TEMPLATES
        #    30% lấy từ TRAIN_TEMPLATES nhưng tổ hợp từ vựng mới hoàn toàn
        n_unseen_tpl = int(TEST_SEEDS_PER_INTENT * 0.7)
        n_shared_tpl = TEST_SEEDS_PER_INTENT - n_unseen_tpl

        test_seeds_unseen = build_seed_pool(
            TEST_TEMPLATES[intent], n_unseen_tpl, global_seen_seeds
        )
        for s in test_seeds_unseen:
            global_seen_seeds.add(remove_vn_diacritics(s.lower()))

        test_seeds_shared = build_seed_pool(
            TRAIN_TEMPLATES[intent], n_shared_tpl, global_seen_seeds
        )
        for s in test_seeds_shared:
            global_seen_seeds.add(remove_vn_diacritics(s.lower()))

        test_seeds = test_seeds_unseen + test_seeds_shared

        # 3. CHIA XONG RỒI MỚI AUGMENT (Nhân 4 biến thể độc lập trên từng tập)
        for seed in train_seeds:
            for variant in generate_4_variants(seed):
                train_dataset.append({"text": variant, "intent": intent})

        for seed in test_seeds:
            for variant in generate_4_variants(seed):
                test_dataset.append({"text": variant, "intent": intent})

        print(
            f"  ✔ Intent [{intent:<8}]: "
            f"Train = {len(train_seeds)*4} mẫu ({len(train_seeds)} seeds) | "
            f"Test = {len(test_seeds)*4} mẫu ({len(test_seeds)} seeds)"
        )

    # Xáo trộn ngẫu nhiên thứ tự mẫu trong từng tập
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    # Lưu các file JSON
    with open(DATA_DIR / "train_8k.json", "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=2)

    with open(DATA_DIR / "test_2k.json", "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=2)

    # Lưu file tổng thể 10k mẫu
    full_10k = train_dataset + test_dataset
    with open(DATA_DIR / "dataset_full.json", "w", encoding="utf-8") as f:
        json.dump(full_10k, f, ensure_ascii=False, indent=2)

    print(f"\n📦 Đã lưu JSON: data/train_8k.json ({len(train_dataset)} mẫu) & data/test_2k.json ({len(test_dataset)} mẫu)")

    # 5. VECTOR HÓA TF-IDF (CHỈ FIT TRÊN TRAIN) & XUẤT RA DATA/PROCESSED/
    print("⚙️ Đang fit TF-IDF (max_features=2000, ngram_range=(1,2)) CHỈ trên tập Train 8.000 mẫu...")
    X_train_texts = [item["text"] for item in train_dataset]
    y_train = np.array([INTENT_TO_ID[item["intent"]] for item in train_dataset])

    X_test_texts = [item["text"] for item in test_dataset]
    y_test = np.array([INTENT_TO_ID[item["intent"]] for item in test_dataset])

    # Giữ max_features=2000 để tương thích 100% với nn.Linear(2000, 512) của PyTorch
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_sparse = vectorizer.fit_transform(X_train_texts)
    X_test_sparse = vectorizer.transform(X_test_texts)

    with open(PROCESSED_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    sp.save_npz(PROCESSED_DIR / "X_train.npz", X_train_sparse)
    sp.save_npz(PROCESSED_DIR / "X_test.npz", X_test_sparse)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    print(f"✅ Đã xuất ma trận Sparse chuẩn vào: {PROCESSED_DIR}")
    print(f"   🔹 X_train shape: {X_train_sparse.shape} | y_train: {y_train.shape}")
    print(f"   🔸 X_test shape:  {X_test_sparse.shape} | y_test:  {y_test.shape}")


if __name__ == "__main__":
    main()
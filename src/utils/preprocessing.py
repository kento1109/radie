import re
import unicodedata
from itertools import permutations

import mojimoji

def zen_to_han(text):
    text = re.sub('〜', '~', text)  # 「〜」を統一
    # NFKC
    return unicodedata.normalize('NFKC', text)

def han_to_zen(text):
    text = mojimoji.han_to_zen(text, ascii=False)
    text= text.replace('#', '＃')
    text= text.replace('(', '（')
    text= text.replace(')', '）')
    text= text.replace('{', '｛')
    text= text.replace('}', '｝')
    text= text.replace(',', '，')
    text= text.replace('.', '．')
    text= text.replace('/', '／')
    text = text.replace(':', '：')
    return text

def remove_header(x):
    ptn_list = ['CHEST', 'Chest CT\r\n', '《 CHEST:CT 》', '検査種CT', '(\(|<|\[)胸部CT(\)|>|\])', '胸部CT:', '胸部CT\r\n', '〔胸部CT:単純〕',
                '胸部.{1,10}CT', '胸腹部.{1,10}CT', '胸腹部同時', '胸部造影\r\n', '頸・胸部CT:', '#plain', '#胸部単純CT', '胸部単純CT', '頸胸部:単純CT',
                '<胸部CT 造影>', '頸胸部:単純CT', 'NECK.*CT', '<NECK>', '<頸部CT>', '頚部.{1,10}CT', '腹部.{1,10}CT', 'CHSTCT','Chest :',
                '単純>', 'CT 》', '〔腹部CT〕', '\[腹部CT：単純\]', '\[腹部CT\]', '［腹部CT］', '\(腹部CT\)',
                '\[腹部CT:単純/造影\]', '腹部CT（単純\+造影）', '\(Abdominal CT\)', '<腹部CT>',

                ]
    # results_list = [re.search(ptn, x, re.IGNORECASE) for ptn in ptn_list]
    results_list = [re.finditer(ptn, x, re.IGNORECASE) for ptn in ptn_list]
    results_list = [[res.end() for res in ptn] for ptn in results_list]
    # results_list = list(filter(lambda result: result is not None, results_list))
    results_list = list(filter(lambda result: len(result) >= 1 , results_list))
    results_list = [max(result) for result in results_list]
    if not results_list:
        return x
    else:
        # s_idx = max([result.end() for result in results_list]) # 最後に出現した位置を取得
        s_idx = max(results_list)  # 最後に出現した位置を取得
        return x[s_idx:]


def remove_footer(x):
    ptn_list = ["一次読影", "読影", "診断医", "影医", "検査医", "検査担当医", "施行医"]
    results_list = [re.search(ptn, x, re.IGNORECASE) for ptn in ptn_list]
    results_list = list(filter(lambda result: result is not None, results_list))
    if not results_list:
        return x
    else:
        e_idx = min([result.start() for result in results_list]) # 最初に出現した位置を取得
        return x[:e_idx]

def remove_char(x):
    # 先頭の削除する文字列の組み合わせ
    header_chars = [':','。', ' ', '>', '】', '》', '\)', '\]', '〉', '≫', '\r\n']
    regex_header_chars = r'^(' + '|'.join([''.join(c) for c in permutations(header_chars, 2)]) + ')'
    x = re.sub(regex_header_chars, '', x)
    regex_header_chars = r'^(' + '|'.join([c for c in header_chars]) + ')'
    x = re.sub(regex_header_chars, '', x)
    x = re.sub(r'《 CHEST:CT 》', '', x)
    regex_char = r'(^-CT>|^CT>|^;CT>|^; CT>|^CT》|-CT》|^CT:)'
    x = re.sub(regex_char, '', x)
    regex_char = r'\((Se\d( |, |、)?)?Im\d{1,3}(-\d{1,3})?\)'
    x = re.sub(regex_char, '', x, re.IGNORECASE)
    regex_char = r'\(Im\d{1,3}Se\d{1,3}\)'
    x = re.sub(regex_char, '', x, re.IGNORECASE)
    regex_char = r'\(Se:?\d{1,3}(,|;|/| )?Im:?\d{1,3}(-\d{1,3})?\)'
    x = re.sub(regex_char, '', x, re.IGNORECASE)
    regex_char = r'\((im|im:|Im:|Im)\d{1,3}(・\d{1,3})?\)'
    x = re.sub(regex_char, '', x, re.IGNORECASE)
    regex_char = r'#(CT|TQ).*#Range:'
    x = re.sub(regex_char, '', x, re.IGNORECASE)
    x = re.sub('。+', '。', x)
    x = re.sub('^ +。', '', x)
    x = re.sub('<胸部CT>', '', x)
    x = re.sub('胸部:。', '', x)
    x = re.sub('Imp[.):].*', '', x)
    return x.strip()

def insert_sep(x):
    # 文の境界を明示
    marker = '[SEP]'
    x = re.sub(r'\r\n\r\n', '\r\n', x)
    x = re.sub(r'。\)', '。', x)
    x = re.sub(r'。\r\n', '。', x)
    # x = re.sub(r'(。|。\))?(\r\n|\r|\n)', marker, x)
    x = re.sub(r'(\r\n|\r|\n)', marker, x)
    x = re.sub('。', marker, x)
    # x = re.sub(r',。', '。', x)  # 「｡」に統一
    x = re.sub(r'(\r\n|\r|\n)', '', x)
    return x

def replace_space(x):
    # a sunny day -> a_sunny_day
    regex_char = r'(?P<char_head>[A-Za-z]) (?P<char_tail>[A-Za-z])'
    x = re.sub(regex_char, "\g<char_head>_\g<char_tail>", x)
    x = re.sub(" ", "", x)
    return x.strip()

def mask_date(x):
    regex_ymd = r"\d{2,4}[年/.-]\d{1,2}[月/.-]\d{1,2}([日/.]+)?"
    x = re.sub(regex_ymd, '＠', x)
    regex_ymd = r"前回.{0,4}\(\d{2}[年/.-]\d{1,2}[月/.-]\d{1,2}([日/.]+)?\)"
    x = re.sub(regex_ymd, '＠', x)
    regex_ym = r"\d{4}[年/.-]\d{1,2}[頃月/.-]"
    x = re.sub(regex_ym, '＠', x)
    regex_y = r"\d{4}年"
    x = re.sub(regex_y, '＠', x)
    regex_md = r"前回\d{1,2}[月/.-]\d{1,2}([日/.]+)?"
    x = re.sub(regex_md, '＠', x)
    regex_md = r"\d{1,2}月\d{1,2}日?"
    x = re.sub(regex_md, '＠', x)
    regex_m = r"\d{1,2}月"
    x = re.sub(regex_m, '＠', x)
    return x

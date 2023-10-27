# # input_string = "8IMM COMP B C'TG M374 50-15A20'S 소전 OI^"

# # # 문자열을 공백을 기준으로 나누고 작은따옴표를 제거하여 리스트 생성
# # result_list = input_string.split()

# # print(result_list)


# a = ['8IMM', 'COMP', 'B', "C'TG", 'M374', "50-15A20'S", '소전', 'OI^']
# b = ['80mm', 'COMN', 'B']

# similar_elements = {}

# for element_b in b:
#     similar_elements[element_b] = []
#     for element_a in a:
#         if element_b in element_a or element_a in element_b:
#             similar_elements[element_b].append(element_a)

# for key, value in similar_elements.items():
#     print(f"Similar elements for {key}: {value}")


# from difflib import SequenceMatcher
# from hangul_romanize import Transliter
# from hangul_romanize.rule import academic

# def text_matching_rate(std, target):
#     transliter = Transliter(academic)
#     jamo_std = transliter.translit(std)
#     jamo_target = transliter.translit(target)
#     match_rate = SequenceMatcher(None, jamo_std, jamo_target).ratio()
#     return match_rate

# std = "가나다"
# target_list = ["가나라", "가마나", "라마바"]

# matching_dict = {}
# for target in target_list:
#     rate = text_matching_rate(std, target)
#     if rate > 0.1:  # 매치율이 0.7보다 높은 경우에만 딕셔너리에 추가
#         matching_dict[target] = rate

# print(matching_dict)


# from nltk.translate.bleu_score import sentence_bleu

# # 예시 문장
# reference = 'this is a sample sentence'
# candidate = 'this is a sample sentence'

# # 문장의 유사도 측정
# score = sentence_bleu([reference.split()], candidate.split())

# print(f"The accuracy score is: {score}")
from nltk.translate.bleu_score import sentence_bleu
reference = [
    'this is a dog'.split(),
    'it is dog'.split(),
    'dog it is'.split(),
    'a dog, it is'.split() 
]
candidate = 'it is dog'.split()
print('BLEU score -> {}'.format(sentence_bleu(reference, candidate )))

candidate = 'it is a dog'.split()
print('BLEU score -> {}'.format(sentence_bleu(reference, candidate)))
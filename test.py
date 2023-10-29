# # # input_string = "8IMM COMP B C'TG M374 50-15A20'S 소전 OI^"

# # # # 문자열을 공백을 기준으로 나누고 작은따옴표를 제거하여 리스트 생성
# # # result_list = input_string.split()

# # # print(result_list)


# # a = ['8IMM', 'COMP', 'B', "C'TG", 'M374', "50-15A20'S", '소전', 'OI^']
# # b = ['80mm', 'COMN', 'B']

# # similar_elements = {}

# # for element_b in b:
# #     similar_elements[element_b] = []
# #     for element_a in a:
# #         if element_b in element_a or element_a in element_b:
# #             similar_elements[element_b].append(element_a)

# # for key, value in similar_elements.items():
# #     print(f"Similar elements for {key}: {value}")





# # from nltk.translate.bleu_score import sentence_bleu

# # # 예시 문장
# # reference = 'this is a sample sentence'
# # candidate = 'this is a sample sentence'

# # # 문장의 유사도 측정
# # score = sentence_bleu([reference.split()], candidate.split())

# # print(f"The accuracy score is: {score}")


# # from nltk.translate.bleu_score import sentence_bleu
# # reference = [
# #     'this is a dog'.split(),
# #     'it is dog'.split(),
# #     'dog it is'.split(),
# #     'a dog, it is'.split() 
# # ]
# # candidate = 'it is dog'.split()
# # print('BLEU score -> {}'.format(sentence_bleu(reference, candidate )))

# # candidate = 'it is a dog'.split()
# # print('BLEU score -> {}'.format(sentence_bleu(reference, candidate)))

# # def calculate_individual_matching_accuracy(list1, list2):
# #     total_elements = len(list2)
# #     match_count = 0
# #     for item1, item2 in zip(list1, list2):
# #         if item1 == item2:
# #             match_count += 1
# #     accuracy = (match_count / total_elements) * 100
# #     return accuracy

# # # 리스트와 기준 문자를 정의합니다
# # my_list = ['ㅁ', 'ㄴ', 'ㄹ', 'ㅎ']
# # reference_list = ['ㅂ', 'ㅁ', 'ㄹ']

# # # 함수를 호출하여 각 요소별 일치하는 정확도를 계산합니다
# # accuracy = calculate_individual_matching_accuracy(my_list, reference_list)
# # print(f"각 요소별 일치하는 정확도: {accuracy}%")


# def find_word(std_list, text_list, threshold=0.5):
#     matching_dict = {}
    
#     for std in std_list:    
#         for text in text_list:
#             rate = text_matching_rate(std, text)
#             if rate >= threshold:
#                 matching_dict[text] = rate
                
#     print(matching_dict)
    
#     return matching_dict


from infer.POCR.ocr_utils import *
def infer_12(ocr_res = "81MM 고'탄 TG M374"):
    std = "81MM COMP B 고폭탄 M374"

    text_list = extract_text(std, ocr_res)
    # text_list = [word for word in text_list.split()]

    print("Processed Text : ", text_list)
    std_list = [word for word in std.split()]
    
    matching_dict = {}
    
    if matching_dict: 
        matching_dict = find_word(std_list, text_list)
    
        print(matching_dict)
        max_key = max(matching_dict, key=matching_dict.get)
        print(max_key)
    
        # check
        if max_key in std_list:
            pass
infer_12()
    
# import re

# def process_string(s):
#     # 정규식 패턴을 사용하여 단어를 찾고, 특정 문자를 제거합니다
#     pattern = r"[A-Za-z0-9가-힣]+"
#     processed_list = re.findall(pattern, s)
#     return processed_list

# # 리스트를 정의합니다
# my_list = ["8 M'M COMP B", "KM374고독탄 로트 트화d9 >라2( 5-032 금야~'"]

# # 각 문자열에 대해 함수를 적용하여 결과를 얻습니다
# result = [process_string(s) for s in my_list]
# print(result)

# a = "Aa가12@"
# b = a.lower()
# print(b)
        
        
    
    
            
# from difflib import SequenceMatcher
# from hangul_romanize import Transliter
# from hangul_romanize.rule import academic

# def greek_text_matching_rate(std, target):
#     transliter = Transliter(academic)
#     jamo_std = transliter.translit(std)
    
#     jamo_target = transliter.translit(target)
    
#     print(jamo_std, jamo_target)
#     match_rate = SequenceMatcher(None, jamo_std, jamo_target).ratio()
#     return match_rate

# std = "고폭 탄"
# target_list = ["81MM", "COMP", "고폭단"]

# matching_dict = {}
# for target in target_list:
#     rate = greek_text_matching_rate(std, target)
#     if rate > 0.1:  # 매치율이 0.7보다 높은 경우에만 딕셔너리에 추가
#         matching_dict[target] = rate

# print(matching_dict)

# from jamo import h2j, j2hcj

# def text_matching_rate(std, target):
#     jamo_std = j2hcj(h2j(std))
#     jamo_target = j2hcj(h2j(target))
#     print(jamo_std, jamo_target)
#     match_rate = SequenceMatcher(None, jamo_std, jamo_target).ratio()
#     return match_rate

# matching_dict = {}
# for target in target_list:
#     rate = text_matching_rate(std, target)
#     if rate > 0.1:  # 매치율이 0.7보다 높은 경우에만 딕셔너리에 추가
#         matching_dict[target] = rate

# print(matching_dict)
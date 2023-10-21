res_info = {"cls": [["a"], [], ["a"], [], ["a", "a"], []], "masks": [[], [], [], [], [], []]}

# 'a'의 위치와 개수를 저장할 빈 리스트 생성
positions = []
count = 0

for row, sublist in enumerate(res_info["cls"]):
    for col, subelement in enumerate(sublist):
        if subelement == 'a':
            positions.append((count, (row, col)))
    count += 1

# 'a'의 위치와 개수 출력
for pos in positions:
    row, col = pos[1]
    print(f"'a'가 위치한 행: {pos[0]}, 열: {col}")

a_count = len(positions)
print(f"'a'의 총 개수: {a_count}")

# 'a'의 위치에 해당하는 masks 리스트의 값을 찾기
print(positions)
for pos in positions:
    row, col = pos
    print(pos)
    # mask_value = res_info["masks"][row][col]
    # print(f"'a'의 위치에 해당하는 masks 값: {mask_value}")

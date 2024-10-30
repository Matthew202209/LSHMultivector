def generate_hamming_codes(binary_code, k):
    """
    生成与给定二进制码汉明距离不大于k的所有二进制码。
    :param binary_code: 给定的二进制码
    :param k: 最大汉明距离
    :return: 所有汉明距离不大于k的二进制码集合
    """
    def hamming_distance(b1, b2):
        return sum(c1 != c2 for c1, c2 in zip(b1, b2))

    def generate_codes(code, distance, max_distance, results):
        if distance > max_distance:
            return
        results.add(code)
        for i in range(len(code)):
            new_code = code[:i] + ('1' if code[i] == '0' else '0') + code[i+1:]
            generate_codes(new_code, distance + 1, max_distance, results)

    results = set()
    generate_codes(binary_code, 0, k, results)
    return list(results)

if __name__ == '__main__':

    # 给定的二进制码和k值
    binary_code = '0001010000000'
    k = 2
    # 生成汉明距离不大于k的二进制码
    hamming_codes = list(generate_hamming_codes(binary_code, k))
    print(hamming_codes)
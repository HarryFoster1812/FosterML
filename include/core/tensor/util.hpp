#include <vector>

std::vector<int> concat_vec(const std::vector<int>& a,
                            const std::vector<int>& b) {
    std::vector<int> result = a;
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

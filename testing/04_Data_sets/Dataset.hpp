#include "../../include/Tensor.hpp"
#include <string>
#include <vector>

namespace FosterML {

template <typename T, typename... Ts> class DataSet {
public:
  std::vector<std::string> columnNames;
  std::vector<std::tuple<Ts...>> data;

  void reserve(size_t n) { data.reserve(n); }

  void addRow(const std::tuple<Ts...> &row) { data.push_back(row); }

  void print(int rows = 5) const {
    for (const auto &columnHeading : columnNames) {
      std::cout << columnHeading << "\t";
    }
    std::cout << std::endl;
    for (const auto &row : data) {
      if (rows == 0)
        break;
      std::apply(
          [](const auto &...fields) {
            ((std::cout << fields << "\t"), ...);
            std::cout << "\n";
          },
          row);
      rows--;
    }
  }

  static std::vector<std::string> getColumnNames(const std::string &path);
  static DataSet<Ts...> sample(const DataSet<Ts...> &data, size_t n);
  static DataSet<Ts...> shuffle(const DataSet<Ts...> &data,
                                bool setSeed = false, int seed = 42);

  // train test split
  static std::vector<DataSet<Ts...>> split(const DataSet<Ts...> &data, size_t n,
                                           std::vector<size_t> &percentages);
  // cross-validation creator?

  // Normalision
  static DataSet<Ts...> normaliseZScore(const DataSet<Ts...> &data);
  static DataSet<Ts...> normaliseMinMax(const DataSet<Ts...> &data);

  size_t numRows() const { return data.size(); }
  size_t numCols() const { return columnNames.size(); }

  DataSet<Ts...> dropMissing() const;
  DataSet<Ts...> fillMissing(T value) const;
  size_t countMissing() const;
  DataSet<Ts...> dropColumns(const std::vector<std::string> &cols) const;
  DataSet<Ts...> dropColumns(const std::vector<int> &colsIndex) const;

  TensorPtr<T> createTensor() const;
};

} // namespace FosterML

#include "Dataset.hpp"
namespace FosterML {

template <typename T> class DataLoader {
  static DataSet<T> loadFromCSV(std::string &path, bool hasHeader = true,
                                char delimiter = ',');
  static DataSet<T> loadFromBinary(std::string &path);
  static DataSet<T> loadMNIST(std::string &path);
};
} // namespace FosterML

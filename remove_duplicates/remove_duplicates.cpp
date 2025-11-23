#include <cstddef>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <gtest/gtest.h>
#include <functional>
#include <string>
#include <iostream>
#include <chrono>
#include <boost/dynamic_bitset.hpp>

std::vector<int> removeDuplicatesReference(const std::vector<int>& arr) {
    if (arr.empty()) {
        return {};
    }
    std::vector<int> sorted = arr;
    std::sort(sorted.begin(), sorted.end());
    auto last = std::unique(sorted.begin(), sorted.end());
    sorted.erase(last, sorted.end());
    return sorted;
}
    

// Algorithm 1: Using std::set
std::vector<int> removeDuplicatesSet(const std::vector<int>& arr) {
    std::set<int> s; 
    for (int num : arr) {
        s.insert(num);
    }
    std::vector<int> result(s.begin(), s.end());
    return result;
}

// Algorithm 2: Using std::unordered_set
std::vector<int> removeDuplicatesUnorderedSet(const std::vector<int>& arr) {
    std::unordered_set<int> s;
    for (int num : arr) {
        s.insert(num);
    }
    std::vector<int> result(s.begin(), s.end());
    return result;
}

// Algorithm 3: Sort and remove adjacent duplicates
std::vector<int> removeDuplicatesSort(const std::vector<int>& arr) {
    if (arr.empty()) {
        return {};
    }

    std::vector<int> sorted = arr;
    std::sort(sorted.begin(), sorted.end());

    std::vector<int> deduped;
    deduped.reserve(sorted.size());
    deduped.push_back(sorted.front());

    for (size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i] != sorted[i - 1]) {
            deduped.push_back(sorted[i]);
        }
    }

    return deduped;
}

// Algorithm 4: Using std::unique after sorting
std::vector<int> removeDuplicatesUnique(const std::vector<int>& arr) {
    if (arr.empty()) {
        return {};
    }

    std::vector<int> sorted = arr;
    std::sort(sorted.begin(), sorted.end());
    auto last = std::unique(sorted.begin(), sorted.end());
    sorted.erase(last, sorted.end());

    return sorted;
}

// Algorithm 5: Using bitset
std::vector<int> removeDuplicatesUsingBitset(const std::vector<int>& arr) {
    if (arr.empty()) {
        return {};
    }

    auto minmax = std::minmax_element(arr.begin(), arr.end());
    const long long minVal = static_cast<long long>(*minmax.first);
    const long long maxVal = static_cast<long long>(*minmax.second);
    const std::size_t range =
        static_cast<std::size_t>(maxVal - minVal + 1);

    boost::dynamic_bitset<std::size_t> bs(range);
    for (int num : arr) {
        const std::size_t idx =
            static_cast<std::size_t>(static_cast<long long>(num) - minVal);
        bs.set(idx);
    }

    std::vector<int> result;
    result.reserve(range);
    for (std::size_t i = 0; i < range; ++i) {
        if (bs.test(i)) {
            result.push_back(static_cast<int>(minVal + static_cast<long long>(i)));
        }
    }

    return result;
}

// Helper function to generate test data
std::vector<int> generateTestData(size_t size, int maxValue = 1000000) {
    std::vector<int> data;
    data.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        data.push_back(rand() % maxValue);
    }
    return data;
}

//////////// Functional Testing ///////////////

const size_t SMALL_SIZE = 100000;
using RemoveDuplicatesFunc = std::function<std::vector<int>(const std::vector<int>&)>;

// Global list of all implementations
struct Implementation {
    std::string displayName;  // For performance tests and output
    std::string testName;     // For Google Test parameter names
    RemoveDuplicatesFunc func;
};

const std::vector<Implementation> ALL_IMPLEMENTATIONS = {
    {"(Set)", "removeDuplicatesSet", removeDuplicatesSet},
    {"(Unordered_Set)", "removeDuplicatesUnorderedSet", removeDuplicatesUnorderedSet},
    {"(Sort_Adjacent)", "removeDuplicatesSort", removeDuplicatesSort},
    {"(Sort_Unique)", "removeDuplicatesUnique", removeDuplicatesUnique},
    {"(Bitset)", "removeDuplicatesBitset", removeDuplicatesUsingBitset}
};

struct TestParam {
    std::string name;
    RemoveDuplicatesFunc func;
};

class RemoveDuplicatesImplTest : public ::testing::TestWithParam<TestParam> {};

TEST_P(RemoveDuplicatesImplTest, MatchesReferenceForAllInputs) {
    const TestParam& impl = GetParam();
    auto testData = generateTestData(SMALL_SIZE);
    auto expected = removeDuplicatesReference(testData);
    auto output = impl.func(testData);
    
    // Normalize both results for comparison (sort them)
    std::sort(expected.begin(), expected.end());
    std::sort(output.begin(), output.end());
    
    EXPECT_EQ(expected, output) << "Implementation: " << impl.name;
}

// Helper function to convert global implementations to test parameters
std::vector<TestParam> getTestParams() {
    std::vector<TestParam> params;
    for (const auto& impl : ALL_IMPLEMENTATIONS) {
        params.push_back({impl.testName, impl.func});
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(
    AllImplementations,        // Instance name
    RemoveDuplicatesImplTest,               // Test case
    ::testing::ValuesIn(getTestParams()),
    [](const ::testing::TestParamInfo<TestParam>& info) {
        // Just use the impl name as the test suffix: e.g. MatchesReferenceForAllInputs/impl1
        return info.param.name;
    }
);

////////// Performance test for large arrays ///////////
using PerfFuncReturnType = std::unordered_map<std::string, double>;
PerfFuncReturnType performanceTestWithLargeArrays(const size_t array_size) {
    auto testData = generateTestData(array_size);

    PerfFuncReturnType durations;

    for(const auto& impl : ALL_IMPLEMENTATIONS) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = impl.func(testData);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        durations[impl.displayName] = duration.count();
    }

    return durations;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    auto c = RUN_ALL_TESTS();

    const size_t MAX_SIZE = 100000000; 

    // Build implementation names list from global list
    std::vector<std::string> impl_names;
    for (const auto& impl : ALL_IMPLEMENTATIONS) {
        impl_names.push_back(impl.displayName);
    }

    std::cout << "ArraySize";
    for (const auto& name : impl_names) {
        std::cout << "\t" << name;
    }
    std::cout << std::endl;

    for (size_t n = 100000; n <= MAX_SIZE; n *= 10) {
        std::cout << n;
        PerfFuncReturnType results = performanceTestWithLargeArrays(n);
        for (const auto& name : impl_names) {
            auto it = results.find(name);
            if (it != results.end()) {
                std::cout << '\t' << it->second << " ms";
            } else {
                std::cout << "\tN/A";
            }
        }
        std::cout << std::endl;
    }
    return 0;
}


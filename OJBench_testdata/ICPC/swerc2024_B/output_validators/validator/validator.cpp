
#include<cstdlib>
int qwerty_getcode(int code) {
    return code == 42? 0: 1;
}
namespace std {
    void qwerty_exit(int code){
        exit(qwerty_getcode(code));
    }
} using std::qwerty_exit;
#define exit qwerty_exit
#define main qwerty_main
// usage: ./a.out input_file correct_output output_dir < contestants_output
// See specification there:
// http://www.problemarchive.org/wiki/index.php/Output_validator

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include "validate.h"
#include <inttypes.h>
#include <algorithm>
#include <cassert>
#include <set>

using namespace std;
int main(int argc, char **argv) {
    init_io(argc, argv);
    
    int N;
    int K;
    judge_in >> N;
    judge_in >> K;
    vector<int64_t> D;
    for (int i = 0; i < N; i++) {
      int64_t d;
      judge_in >> d;
      D.push_back(d);
    }
    assert(N > 0);
    const int64_t max_input_day = *max_element(D.begin(), D.end());
    int64_t judge_cost = 0;
    for (int i = 0; i < N; i++) {
      int64_t d;
      judge_ans >> d;
      assert(d >= D[i]);
      judge_cost += (d - D[i]) * (d - D[i]);
    }
    int64_t author_cost = 0;
    set<int64_t> author_days;
    for (int i = 0; i < N; i++) {
      int64_t d;
      if (!(author_out >> d)) {
	wrong_answer("Wrong answer (probably too short at index %d)\n", i);
      }
      if (d < 0) {
	wrong_answer("Wrong answer: negative day %" PRId64 "\n", d);
      }
      if (d < D[i]) {
	wrong_answer("Wrong answer: delivering gift too early %" PRId64 " < %" PRId64 "\n",
		     d, D[i]);
      }
      if (d > max_input_day) {
	// This prevents a possible overflow when computing (d - D[i])^2 below.
	wrong_answer("Wrong answer: delivering gift in day %" PRId64 " > %" PRId64 " is never optimal\n",
		     d, max_input_day);
      }
      author_days.insert(d);
      author_cost += (d - D[i]) * (d - D[i]);
    }

    if (author_days.size() > K) {
      wrong_answer("Invalid answer: Too many different days used");
    }
    
    if (author_cost != judge_cost) {
      wrong_answer("Invalid answer: author cost %" PRId64 " != judge cost %" PRId64 "\n",
		   author_cost, judge_cost);
    }
    accept();
}

#undef main
#include<cstdio>
#include<vector>
#include<string>
#include<filesystem>
int main(int argc, char **argv) {
	namespace fs = std::filesystem;
    freopen(argv[2], "r", stdin);
    char judge_out[] = "/dev";
    std::vector<char*> new_argv = {
		argv[0], argv[1], argv[3],
		judge_out,
	};
	return qwerty_getcode(qwerty_main((int)new_argv.size(), new_argv.data()));
}


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
#include "validation.h"

const int MAX_Q = 1000;
const int EXTRA_Q = 2000;

int query(std::vector<std::vector<int>>& f, int c, int r) {
	r--;
	int k = 0;
	while(c > 0) {
		if(c % 2 == 1) r = f[k][r];
		c /= 2;
		k++;
	}
	r++;
	return r;
}

int main(int argc, char* argv[]) {
	// Set up the input and answer streams.
	std::ifstream in(argv[1]);
	// std::ifstream ans(argv[2]); // Only for custom checker.
	OutputValidator v(argc, argv);

	int n;
	in >> n;
	std::cout << n << std::endl;

	std::vector<std::vector<int>> f(30, std::vector<int>(n)); // f[i][j] = f^(2^i)(j)
	for(int i = 0; i < n; i++) {
		in >> f[0][i];
		f[0][i]--;
	}
	for(int i = 1; i < 30; i++) {
		for(int j = 0; j < n; j++) {
			f[i][j] = f[i - 1][f[i - 1][j]];
		}
	}

	int used_q = 0;
	while(true) {
		std::string q = v.test_strings({"?", "!"});
		if(q == "?") {
			used_q++;
			v.check(used_q <= EXTRA_Q, "Used more than ", EXTRA_Q, " queries, aborting.");
			v.space();
			int c = v.read_integer("c", 1, n);
			v.space();
			int r = v.read_integer("r", 1, n);
			v.newline();
			std::cout << query(f, c, r) << std::endl;
		} else {
			v.space();
			int c = v.read_integer("c", 1, n);
			v.space();
			int r = v.read_integer("r", 1, n);
			v.newline();
			std::cerr << "Used " << used_q << " queries. "; // No endl, so that a possible WA message ends up on the same line.
			v.check(query(f, c, r) == c, "Answer is wrong.");
			v.check(used_q <= MAX_Q, "Used more than ", MAX_Q, " queries.");
			break;
		}
	}
}

#undef main
#include<cstdio>
#include<vector>
#include<string>
#include<filesystem>
int main(int argc, char **argv) {
	namespace fs = std::filesystem;
    char judge_out[] = "/dev";
    std::vector<char*> new_argv = {
		argv[0], argv[1], argv[2],
		judge_out,
	};
	return qwerty_getcode(qwerty_main((int)new_argv.size(), new_argv.data()));
}

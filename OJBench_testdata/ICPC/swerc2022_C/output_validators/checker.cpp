
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
#include "testlib.h"

#include <string>
#include <vector>

int main(int argc, char **argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    std::string s = inf.readWord();

    std::vector<int> prefix_sums(2 * n, 0);
    for (int i = 0; i < 2 * n - 1; ++i) {
        prefix_sums[i + 1] = prefix_sums[i] + int(s[i] == 'W');
    }

    int x = ouf.readInt(0, 2 * n - 1, "x");
    
    int cnt = 0;
    int l = 0, r;
    for (r = 0; r < 2 * n - 1 && prefix_sums[r + 1] < x; ++r);
    while (cnt < n && r < 2 * n - 1) {
        while (prefix_sums[r + 1] - prefix_sums[l] > x) ++l;
        for (
                int i = l;
                cnt < n
                    && r - i + 1 >= n
                    && prefix_sums[r + 1] - prefix_sums[i] == x;
                ++i
            ) { ++cnt; }
        ++r;
    }

    if (cnt == n) quitf(_ok, "At least n intervals");
    else quitf(_wa, "Not enough intervals");

    return 0;
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

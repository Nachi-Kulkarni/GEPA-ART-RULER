
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

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

int main(int argc, char **argv) {
    registerInteraction(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    std::vector<int> x = inf.readInts(n);

    std::cout << n << " " << m << std::endl;
    for (int i = 0; i < n - 1; ++i) std::cout << x[i] << " ";
    std::cout << x[n - 1] << std::endl;

    std::string player = ouf.readWord();

    std::vector<int> p(n);
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](int const& a, int const& b) {
        return x[a] > x[b];
    });

    bool win;

    if (player == "Bernardo") {         // Play as Alessia
        std::vector<int> chosen(m + 1, 0);
        for (int t = 0; t < n; ++t) {
            int y = x[p[t]];
            int last = 0;
            int min_width = m + 1;
            int a = 0;
            for (int i = 1; i <= m; ++i) {
                if (chosen[i]) {
                    last = i;
                }
                if ((i == m || chosen[i + 1]) && i - last >= y && i - last < min_width) {
                    min_width = i - last;
                    a = last + 1;
                }
            }
            if (a == 0) a = rnd.next(1, m - y + 1);
            std::cout << y << " " << a << std::endl;
            int b = ouf.readInt(a, a + y - 1);
            ++chosen[b];
        }
        win = true;
        for (int i = 1; i <= m; ++i) {
            if (chosen[i] >= 2) {
                win = false;
                break;
            }
        }
    } else if (player == "Alessia") {   // Play as Bernardo
        int k = 0;
        for (int j = 1; j < n; ++j) {
            if (m / (j + 1) - x[p[j]] < m / (k + 1) - x[p[k]]) {
                k = j;
            }
        }
        std::vector<int> chosen(m + 1, 0);
        std::vector<int> remaining(m + 1, 0);
        for (int y : x) ++remaining[y];
        for (int t = 0; t < n; ++t) {
            int y = ouf.readInt(1, m);
            if (!remaining[y]) quitf(_wa, "Chosen an invalid y");
            --remaining[y];
            int a = ouf.readInt(1, m - y + 1);
            int b = 0;
            for (int i = a; i <= a + y - 1; ++i) {
                if (i % x[p[k]] == 0) {
                    b = i;
                    break;
                }
            }
            if (b == 0) b = rnd.next(a, a + y - 1);
            ++chosen[b];
            std::cout << b << std::endl;
        }
        win = false;
        for (int i = 1; i <= m; ++i) {
            if (chosen[i] >= 2) {
                win = true;
                break;
            }
        }
    } else quitf(_pe, "Invalid player name");

    if (win) quitf(_wa, "Judge won the game");
    else quitf(_ok, "Contestant won the game");

    return 0;
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
		argv[0], argv[1], "/dev/null", argv[2],
		judge_out,
	};
	return qwerty_getcode(qwerty_main((int)new_argv.size(), new_argv.data()));
}

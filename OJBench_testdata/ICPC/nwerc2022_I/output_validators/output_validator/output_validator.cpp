
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

int main(int argc, char **argv) {
    std::ifstream in(argv[1]);
    OutputValidator v(argc, argv);

    int c, d;
    in >> c >> d;

    std::vector<std::string> items(d-c+1);
    for (auto &s: items) in >> s;

    int a = v.read_integer("a", 1, 1'000'000);
    v.space();
    int b = v.read_integer("b", 1, 1'000'000);
    v.newline();

    for (int i = c; i <= d; i++) {
        std::string str;
        if (i%a == 0) str += "Fizz";
        if (i%b == 0) str += "Buzz";
        if (str.empty()) str += std::to_string(i);
        
        v.check(str == items[i-c], "Mismatch at position ", i, ": expected ", items[i-c], ", got ", str);
    }
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

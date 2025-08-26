
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
#include <iostream>
#include <cassert>
#include "validation.h"

std::string wordle_query(std::string query, std::string answer) {
    int n = std::size(query);
    std::string res(n, 'B');

    std::array<int,26> count = {};
    for (char c: answer) count[c-'a']++;

    // place green tiles
    for (int i = 0; i < n; i++) {
        if (query[i] == answer[i]) {
            res[i] = 'G';
            count[query[i]-'a']--;
        } 
    }

    // place yellow tiles
    for (int i = 0; i < n; i++) {
        if (res[i] != 'G' && count[query[i]-'a'] > 0) {
            count[query[i]-'a']--;
            res[i] = 'Y';
        }
    }

    return res;
}

int main(int argc, char **argv) {
    std::ifstream in(argv[1]);
    OutputValidator v(argc, argv);

    int m, n;
    in >> m >> n;

    std::vector<std::string> words(m-1), colors(m-1);
    for (int i = 0; i < m-1; i++) {
        in >> words[i] >> colors[i];
    }
    
    std::string all_letters = std::string(26, 'a') + std::string(26, 'A');
    for (int i = 0; i < 26; i++) {
        all_letters[i] += i;
        all_letters[26+i] += i;
    }
    
    // TODO: extend validation.h with case insensitivity?
    std::string answer = v.read_string("word", n, n, all_letters);
    for (char& c : answer) c |= 0x20;
    v.newline();

    for (int i = 0; i < m-1; i++) {
        v.check(wordle_query(words[i], answer) == colors[i],
                "Wrong answer: does not fit word ", i);
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

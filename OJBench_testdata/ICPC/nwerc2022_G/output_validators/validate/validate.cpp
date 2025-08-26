
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
#include <climits>
#include <set>
#include <vector>

#include "validate.h"
using namespace std;

void to_lower(std::string& s) {
    for (char& c : s) c |= 0x20;
}

int main(int argc, char **argv) {
    init_io(argc,argv);

    int n;
    judge_in >> n;

    int position = 0;
    vector<int> state(n);
    for (int i = 0; i < n; i++) {
        char c;
        judge_in >> c;
        state[i] = c - '0';
    }

    cout << state[position] << endl;

    int max_queries = 3*n + 500;
    int lim_queries = 5*n + 1000;
    int queries_used = 0;
    while (true) {
        string response;
        if (!(author_out >> response)) {
            wrong_answer("Wrong answer: failed to read response. Queries used: %d\n", queries_used);
        }

        if (response == "?") {
            queries_used++;

            string query;
            if (!(author_out >> query)) {
                wrong_answer("Wrong answer: failed to read query. Queries used: %d\n", queries_used);
            }
            to_lower(query);

            if (query == "left") {
                position = (position + n - 1) % n;
            } else if (query == "right") {
                position = (position + 1) % n;
            } else if (query == "flip") {
                state[position] = 1 - state[position];
            } else {
                wrong_answer("Wrong answer: invalid query type. Queries used: %d\n", queries_used);
            }

            cout << state[position] << endl;
        } else if (response == "!") {
            int guess;
            if (!(author_out >> guess)) {
                wrong_answer("Wrong answer: failed to read guess. Queries used: %d\n", queries_used);
            }

            string junk;
            if (author_out >> junk) {
                wrong_answer("Wrong answer: trailing output after guess.\n");
            }

            if (guess == n) {
                if (queries_used > max_queries) {
                    wrong_answer("Wrong answer: correct guess, but used too many queries. Queries used: %d (%d too many)\n", queries_used, queries_used-max_queries);
                }
                break;
            } else {
                wrong_answer("Wrong answer: incorrect guess %d (ans: %d). Queries used: %d\n", guess, n, queries_used);
            }
        } else {
            wrong_answer("Wrong answer: invalid response type. Queries used: %d\n", queries_used);
        }
        if (queries_used > lim_queries) {
            wrong_answer("Wrong answer: used too many queries. Queries used: %d (%d too many, terminated)\n", queries_used, queries_used-max_queries);
        }
    }

    string junk;
    if (author_out >> junk) {
        wrong_answer("Wrong answer: trailing output.\n");
    }

    judge_message("Queries used: %d.\n", queries_used);

    accept();
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

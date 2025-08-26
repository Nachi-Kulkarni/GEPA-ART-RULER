
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
using std::vector;
using std::string;

// Output: a list of `n` page flips from `i` to `j`.

int main(int argc, char **argv) {
    // Set up the input and answer streams.
    std::ifstream in(argv[1]);
    std::ifstream ans(argv[2]); // Only for custom checker.
    OutputValidator v(argc, argv);

    long long a,b;
    char c;
    in >> a >> c >> b;
    string s;
    ans >> s;
    if (s == "impossible"){
        v.test_string("impossible");
        v.newline();
    }
    else {
        long long n = v.read_integer("n",1,1e6);
        v.space();
        long long m = v.read_integer("m",1,1e6);
        v.newline();
        vector<vector<long long>> graph(n+1,vector<long long>{});
        for (long long i=0;i<m;i++){
            long long x = v.read_integer("x", 1, n);
            v.space();
            long long y = v.read_integer("x", 1, n);
            v.newline();
            graph[x].push_back(y);
            graph[y].push_back(x);
        }
        long long total = 0;
        vector<bool> visited(n+1);
        vector<long long> queue_out, queue_in;
        long long distance = 0;
        queue_out.push_back(1);
        while (queue_out.size() > 0){
            for (long long i : queue_out){
                if (visited[i])
                    continue;
                total += distance;
                visited[i] = true;
                for (long long j : graph[i])
                    queue_in.push_back(j);
            }
            queue_out = queue_in;
            queue_in = {};
            distance += 1;
        }
        for (long long i=1;i<=n;i++) v.check(visited[i], "Graph not connected, vertex ", i, " unreachable");
        std::cerr << total << '/' << n << std::endl;
        v.check(a*n == total*b, "Wrong average distance, got ", total, "/", n, ", wanted ", a, "/", b);
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

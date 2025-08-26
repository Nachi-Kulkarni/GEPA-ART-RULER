
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
// This checker was obtained by modifying EPS and error messages in the standard doubles checker.
// I also added "-inf, inf" and variable names to readDouble to appease the Polygon warning.
// I did not want to set 0.0/1.0 as limits since that would make a solution outputting -1e-100 instead of 0
// fail.

#include "testlib.h"
#include <cmath>
#include <limits>
 
using namespace std;
 
#define EPS 1.1e-8
 
string ending(int x)
{
    x %= 100;
    if (x / 10 == 1)
        return "th";
    if (x % 10 == 1)
        return "st";
    if (x % 10 == 2)
        return "nd";
    if (x % 10 == 3)
        return "rd";
    return "th";
}
 
int main(int argc, char * argv[])
{
    setName("compare two sequences of doubles, max absolute or relative  error = %.9lf", EPS);
    registerTestlibCmd(argc, argv);
 
    int n = 0;
    double j, p;
    double max_error = 0;
    double inf = numeric_limits<double>::infinity();
 
    while (!ans.seekEof()) 
    {
      n++;
      j = ans.readDouble(-inf, inf, "answer");
      p = ouf.readDouble(-inf, inf, "output");
      if (!doubleCompare(j, p, EPS))
        quitf(_wa, "%d%s numbers differ - expected: '%.9lf', found: '%.9lf', error = '%.9lf'", n, ending(n).c_str(), j, p, doubleDelta(j, p));
      max_error = max(max_error, abs(j - p));
    }
 
    if (n == 1)
        quitf(_ok, "found '%.9lf', expected '%.9lf', error '%.9lf'", p, j, doubleDelta(j, p));
 
    quitf(_ok, "%d numbers, max absolute error %.9lg", n, max_error);
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

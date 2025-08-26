
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
#include "./testlib.h"
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;
typedef pair<int, int> pii;
typedef long long ll;
#define mp make_pair

const int N = 55;
int n, m, k;
int ed[N * N][2];
ll g[N];
vector<int> col[2 * N];
int q[N];
int topQ;

void addEdges(vector<int> edges) {
	for (int e : edges) {
		int v = ed[e][0], u = ed[e][1];
		g[v] |= 1LL << u;
		g[u] |= 1LL << v;
	}
}

bool isConnected() {
	ll used = 1;
	topQ = 0;
	q[topQ++] = 0;
	for (int i = 0; i < topQ; i++) {
		int v = q[i];
		ll mask = g[v] ^ (g[v] & used);
		while(mask != 0) {
			int u = __builtin_ctzll(mask);
			mask ^= 1LL << u;
			used ^= 1LL << u;
			q[topQ++] = u;
		}
	}
	return topQ == n;
}

void checkTest(InStream &inf, InStream &ouf, int testId) {
    setTestCase(testId);
	n = inf.readInt();
	m = inf.readInt();
	for (int i = 0; i < m; i++)
		for (int j = 0; j < 2; j++) {
			ed[i][j] = inf.readInt();
			ed[i][j]--;
		}
	k = ouf.readInt(-1, m, "k");
	if (k == -1) ouf.quitf(_wa, "answer not found");
	if (k <= 1) ouf.quitf(_wa, "k is too small - %d < 2", k);
	if (k > 2 * n) ouf.quitf(_wa, "k is too big - %d > 2 * %d", k, n);
	for (int i = 0; i < k; i++)
	    col[i].clear();
	vector<int> c = ouf.readInts(m, 1, k, "c");
	for (int i = 0; i < m; i++) {
		col[c[i] - 1].push_back(i);
	}
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < n; j++)
			g[j] = 0;
		addEdges(col[i]);
		if (isConnected()) ouf.quitf(_wa, "color %d is connected", i + 1);
	}
	for (int x = 0; x < k; x++)
		for (int y = x + 1; y < k; y++) {
			for (int j = 0; j < n; j++)
				g[j] = 0;
			addEdges(col[x]);
			addEdges(col[y]);
			if (!isConnected()) ouf.quitf(_wa, "colors %d and %d are not connected", x + 1, y + 1);
		}
}

int main(int argc, char * argv[])
{
    registerTestlibCmd(argc, argv);

    int t = inf.readInt();
    for (int i = 1; i <= t; i++) {
    	checkTest(inf, ouf, i);
    }

    quitf(_ok, "OK");

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


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
#include <iostream>

using namespace std;

using ll = long long;

const ll infty = 1e18;

ll area(ll x1, ll x2, ll x3, ll y1, ll y2, ll y3) {
  return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
}

int find_optimal_position(int n, const vector<ll>& xx, const vector<ll>& yy,
    const vector<bool>& alive) {
  ll mn = infty;
  int opt = -1;
  int prev1 = n - 1;
  while (not alive[prev1]) --prev1;
  int prev2 = prev1 - 1;
  while (not alive[prev2]) --prev2;

  for (int i = 0; i < n; ++i) {
    if (not alive[i]) continue;
    ll a = area(xx[prev2], xx[prev1], xx[i], yy[prev2], yy[prev1], yy[i]);
    if (a < mn) {
      mn = a;
      opt = prev1;
    }
    prev2 = prev1;
    prev1 = i;
  }
  return opt;
}

int find_2nd_optimal_position(int n, const vector<ll>& xx, const vector<ll>& yy,
    const vector<bool>& alive) {
  ll mn1 = infty, mn2 = infty;
  int opt1 = -1, opt2 = -1;

  int prev1 = n - 1;
  while (not alive[prev1]) --prev1;
  int prev2 = prev1 - 1;
  while (not alive[prev2]) --prev2;

  for (int i = 0; i < n; ++i) {
    if (not alive[i]) continue;
    ll a = area(xx[prev2], xx[prev1], xx[i], yy[prev2], yy[prev1], yy[i]);
    if (a < mn1) {
      mn2 = mn1;
      opt2 = opt1;
      mn1 = a;
      opt1 = prev1;
    } else if (a < mn2) {
      mn2 = a;
      opt2 = prev1;
    }
    prev2 = prev1;
    prev1 = i;
  }
  return opt2;
}

int find_random_position(int n, const vector<bool>& alive) {
  vector<int> pos;
  for (int i = 0; i < n; ++i) {
    if (alive[i]) pos.push_back(i);
  }
  int q = pos.size();
  return pos[rnd.next(q)];
}

int find_position(int n, const vector<ll>& xx, const vector<ll>& yy,
    const vector<bool>& alive, int strategy) {
  if (strategy == 0) return find_optimal_position(n, xx, yy, alive);
  if (strategy == 1) return find_2nd_optimal_position(n, xx, yy, alive);
  if (strategy == 2) return find_random_position(n, alive);
  return -1;
}

ll find_area(int p, int n, const vector<ll>& xx, const vector<ll>& yy,
    const vector<bool>& alive) {
  int nxt = (p + 1) % n;
  while (not alive[nxt]) nxt = (nxt + 1) % n;
  int prv = (p + n - 1) % n;
  while (not alive[prv]) prv = (prv + n - 1) % n;
  return area(xx[prv], xx[p], xx[nxt], yy[prv], yy[p], yy[nxt]);
}

int main(int argc, char* argv[]) {
  setName("Greedy Interactor");
  registerInteraction(argc, argv);

  rnd.setSeed(42);

  int n = inf.readInt();
  vector<ll> xx(n), yy(n);

  cout << n << endl;
  for (int i = 0; i < n; ++i) {
    xx[i] = inf.readLong();
    yy[i] = inf.readLong();
    cout << xx[i] << ' ' << yy[i] << endl;
  }

  int strategy = 0;  // optimal
  if (n == 44 or n == 45) {
    strategy = 1;  // second optimal
  }
  if (n == 78 or n == 79) {
    strategy = 2;  // random
  }

  string he = ouf.readToken();
  tout << "< " << he << endl;

  vector<bool> alive(n, true);

  ll our_area = 0, their_area = 0;

  int cur = n;
  bool our_turn;

  if (he == "Alberto") {
    our_turn = false;
  } else if (he == "Beatrice") {
    our_turn = true;
  } else {
    quitf(_wa, "unrecognised player name");
  }

  while (cur > 2) {
    if (our_turn) {
      int p = find_position(n, xx, yy, alive, strategy);
      our_area += find_area(p, n, xx, yy, alive);

      cout << p + 1 << endl;
      tout << "> " << p + 1 << endl;
      alive[p] = false;
    } else {
      int p = ouf.readInt(1, n, "chosen vertex");
      tout << "< " << p << endl;
      --p;
      quitif(!alive[p], _wa, "vertex already chosen");
      their_area += find_area(p, n, xx, yy, alive);
      alive[p] = false;
    }

    our_turn = !our_turn;
    --cur;
  }
  
  if (their_area <= our_area) {
    quitf(_ok, "they win, their_area %lld <= our_area %lld. Used strategy %d", their_area, our_area, strategy);
  } else {
    quitf(_wa, "they lose, their_area %lld > our_area %lld. Used strategy %d", their_area, our_area, strategy);
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
		argv[0], argv[1], "/dev/null", argv[2],
		judge_out,
	};
	return qwerty_getcode(qwerty_main((int)new_argv.size(), new_argv.data()));
}

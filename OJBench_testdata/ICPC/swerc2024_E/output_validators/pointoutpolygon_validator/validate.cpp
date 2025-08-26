
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
#include "validate.h"

#include <complex>
#include <numeric>
#include <set>
#include <vector>
using namespace std;

typedef long long ll;
typedef complex<ll> pt;

ll dot(pt v, pt w) { return (conj(v) * w).real(); }
ll cross(pt v, pt w) { return (conj(v) * w).imag(); }
ll orient(pt a, pt b, pt c) { return cross(b-a, c-a); }

bool inDisk(pt a, pt b, pt p) {
    return dot(a - p, b - p) <= 0;
}

bool onSegment(pt a, pt b, pt p) {
    return orient(a, b, p) == 0 && inDisk(a, b, p);
}

bool intersect(pt a, pt b, pt c, pt d) {
  __int128_t oa = orient(c,d,a), ob = orient(c,d,b),
             oc = orient(a,b,c), od = orient(a,b,d);
  return oa * ob < 0 && oc * od < 0;
}

ll area(vector<pt> P) {
  ll area = 0;
  for (size_t i = 0; i < P.size(); i++) {
    area += cross(P[i], P[(i + 1) % P.size()]);
  }
  return abs(area);
}

int main(int argc, char **argv) {
    init_io(argc, argv);

    size_t N;
    if (!(judge_in >> N)) judge_error("Invalid input");

    vector<pt> points(N);
    for (size_t i = 0; i < N; i++) {
        ll x, y;
        if (!(judge_in >> x >> y)) judge_error("Invalid input");
        points[i] = {x, y};
    }

    /*
      Constraints:
       - [x] K <= 3N
       - [x] polygon is simple and non-degenerate:
         - [x] no two edges intersect
         - [x] each edge does not touch any other edge
         - [x] each vertex is unique
       - [x] the initial points are all vertices of the polygon
       - [x] no integer point is located inside the polygon
       - [x] all vertices coordinates are between 1 and 10^9
    */

    size_t K;
    if (!(author_out >> K)) wrong_answer("Malformed output");
    if (K < 3) wrong_answer("Too few vertices");
    if (K > 3 * N) wrong_answer("Too many vertices");

    vector<pt> polygon(K);
    for (size_t i = 0; i < K; i++) {
        ll x, y;
        if (!(author_out >> x >> y)) wrong_answer("Malformed output");
        polygon[i] = {x, y};
        if (!(1 <= x && x <= 1e9) || !(1 <= y && y <= 1e9)) {
            wrong_answer("Vertex outside the boundary: (%lld, %lld)", x, y);
        }
    }

    auto comp = [](const pt& a, const pt& b) {
        return pair(a.real(), a.imag()) < pair(b.real(), b.imag());
    };

    set<pt, decltype(comp)> vertices(comp);
    for (size_t i = 0; i < K; i++) {
        if (!vertices.insert(polygon[i]).second) {
            wrong_answer("Duplicate vertex: (%lld, %lld)", polygon[i].real(), polygon[i].imag());
        }
    }

    for (size_t i = 0; i < N; i++) {
        if (!vertices.count(points[i])) {
            wrong_answer("Missing points: (%lld, %lld)", points[i].real(), points[i].imag());
        }
    }

    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < K; j++) {
            if (i == j) continue;
            if (intersect(polygon[i], polygon[(i + 1) % K], polygon[j], polygon[(j + 1) % K])) {
                wrong_answer("Intersects itself");
            }

            if ((i + 1) % K == j) continue;
            if (onSegment(polygon[i], polygon[(i + 1) % K], polygon[j])) {
                wrong_answer("Degenerate polygon");
            }
        }
    }

    ll boundary = 0;
    for (size_t i = 0; i < K; i++) {
        pt d = polygon[i] - polygon[(i + 1) % K];
        ll dx = abs(d.real());
        ll dy = abs(d.imag());
        if (dx == 0 || dy == 0) {
            boundary += dx + dy;
        } else {
            boundary += gcd(dx, dy);
        }
    }
    if (boundary - area(polygon) != 2) {
        wrong_answer("Integer points inside");
    }

    string trash;
    if (author_out >> trash) {
        wrong_answer("Malformed output");
    }

    accept();
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

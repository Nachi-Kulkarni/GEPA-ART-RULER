
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
#include <set>
#include <string>
#include <iostream>

#define CARRIER_LENGTH 5
#define MAX_QUERIES 2500

using namespace std;

typedef pair<int, int> point;

// View on the grid for the adaptive validator, which tells us how many
// ships could still be potentially placed, in O(log n) time per update.
class Grid {
public:
  int possibleCarriers;
private:
  const int n;
  // Swap x,y for the symmetry?
  const bool swap;
  // Points that are taken, either because of a miss or because we placed a ship
  vector<set<int>> taken;

public:
  Grid(int n, bool swap): n(n), swap(swap) {
    possibleCarriers = 0;
    for(int i = 1; i <= n; i++) {
      // Insert outside boundaries, this makes checks later on easier
      set<int> row;
      row.insert(0);
      row.insert(n+1);
      taken.push_back(row);
      possibleCarriers += numCarriers(0, n + 1);
    }
  }

  // How many carriers fit between left and right?
  static int numCarriers(int left, int right) {
    return (right - left - 1) / CARRIER_LENGTH;
  }

  // Take out point p, this updates possibleCarriers
  void take(point p) {
    int r = swap ? p.second : p.first;
    int c = swap ? p.first : p.second;

    // Search the earliest value after this point in this row
    auto it = taken[r-1].upper_bound(c);
    int right = *it;
    it--;
    int left = *it;

    // If this point was hit before we don't update anything
    if(left == c) return;

    // Update counts, we now split this interval
    possibleCarriers -= numCarriers(left, right);
    possibleCarriers += numCarriers(left, c);
    possibleCarriers += numCarriers(c, right);

    // And insert the point
    taken[r-1].insert(c);
  }

  const void possiblePlacements(point p, vector<pair<point,bool>>& places) {
    int r = swap ? p.second : p.first;
    int c = swap ? p.first : p.second;

    // Find the left and right, but the current point was already inserted, so ignore it
    auto it = taken[r-1].upper_bound(c);
    int right = *it;
    it--;
    if(*it != c) {
      cerr << "Judge error: cell should already have been shot at: " << p.first << " "
           << p.second << endl;
      exit(1);
    }
    it--;
    int left = *it;

    // Loop over the leftmost point of the carrier
    for(int i = max(1, c - CARRIER_LENGTH + 1); i <= min(n - CARRIER_LENGTH + 1, c); i++) {
      // Check that it fits
      if(left >= i) continue;
      if(right < i + CARRIER_LENGTH) continue;

      // Check that this won't change the amount of possible carriers
      if(numCarriers(left, c) != numCarriers(left, i)) continue;
      if(numCarriers(c, right) != numCarriers(i + CARRIER_LENGTH - 1, right)) continue;

      // It worked, so this is a possible carrier placement
      places.push_back(make_pair(make_pair(swap ? i : r, swap ? r : i), swap));
    }
  }
};

int main(int argc, char *argv[]) {
  // Read the basic inputs
  std::ifstream in(argv[1]);
  OutputValidator v(argc, argv);
  int n, k, f, seed;
  double p;
  in >> n >> k >> f >> seed >> p;
  cout << n << ' ' << k << endl;
  srand(seed);

  // Store the carrier locations
  vector<int> numHits(k, 0);
  map<point,int> carriers;

  // Make two gridviews, one for horizontal and one for vertical
  auto grids = { new Grid(n, false), new Grid(n, true) };

  // Place the fixed carriers
  for (int i = 0; i < f; i++) {
    string dir;
    int x, y;
    in >> dir >> x >> y;
    bool hor = (dir=="h");
    for(int j = 0; j < CARRIER_LENGTH; j++) {
      auto pt = make_pair(x + (hor?1:0)*j, y + (hor?0:1)*j);
      if(carriers.find(pt) != carriers.end()) {
        cerr << "Judge error: Trying to place second carrier at " << pt.first << "," << pt.second << endl;
        exit(1);
      }
      carriers[pt] = i;
      for(auto grid : grids)
        grid->take(pt);
    }
  }
  int carriersRemaining = k - f;

  // Store what we've hit so far
  set<point> hits;
  set<point> shots;

  // Go through the queries
  for (int q = 1; q <= 2 * MAX_QUERIES; q++) {
    // Get and check contestant input
    int x = v.read_integer("x", 1, n);
    v.space();
    int y = v.read_integer("y", 1, n);
    v.newline();
    point shot = make_pair(x, y);
    if (shot.first < 1 || shot.first > n || shot.second < 1 || shot.second > n) {
      v.WA("Shot is out of bounds: ", x, ",", y);
    }

    // Repeated shots are always misses
    if(shots.find(shot) != shots.end()) {
      cout << "miss" << endl;
      continue;
    }
    shots.insert(shot);

    // Update the adaptive interactor, and check if we now need to place a carrier
    if(carriersRemaining > 0 && carriers.find(shot) == carriers.end()) {
      // If this is a miss but then none of the grids an fit the remaining carriers,
      // we have to place it now and call this a hit.
      bool mustPlace = true;
      for(auto grid : grids) {
        grid->take(shot);
        if(grid->possibleCarriers >= carriersRemaining)
          mustPlace = false;
      }

      // Or maybe we randomly hit early
      bool earlyHit = ((double)rand() / (double)RAND_MAX) <= p;

      // If needed we place it
      if(mustPlace || earlyHit) {
        vector<pair<point,bool>> places;
        for(auto grid : grids)
          // Only consider the early hit when this grid allows placing enough carriers
          if((earlyHit && grid->possibleCarriers >= carriersRemaining - 1)
              || grid->possibleCarriers == carriersRemaining - 1)
            grid->possiblePlacements(shot, places);

        // See if we found some places, this could be false when earlyHit = true
        if(places.size() == 0) {
          // Double check, when mustPlace = true this should never happen...
          if(mustPlace) {
            cerr << "Judge error: Cannot find carrier to place containing point " << shot.first << "," << shot.second << endl;
            exit(1);
          }
        } else {
          // Pick one of these at random
          int pick = rand() % places.size();
          int x = places[pick].first.first;
          int y = places[pick].first.second;
          bool hor = places[pick].second;
          cerr << "Placing at " << x << "," << y << " " << (hor?"h":"v") << endl;
          for(int j = 0; j < CARRIER_LENGTH; j++) {
            auto pt = make_pair(x + (hor?1:0)*j, y + (hor?0:1)*j);
            if(carriers.find(pt) != carriers.end()) {
              cerr << "Judge error: Trying to place second carrier at " << pt.first << "," << pt.second << endl;
              exit(1);
            }
            carriers[pt] = k - carriersRemaining;
            for(auto grid : grids)
              grid->take(pt);
          }
          carriersRemaining--;
        }
      }
    }

    // If we did not shoot before and we found a carrier, the shot hits
    if(hits.find(shot) == hits.end() && carriers.find(shot) != carriers.end()) {
      int carrier = carriers.find(shot)->second;
      numHits[carrier]++;
      cout << (numHits[carrier] == CARRIER_LENGTH ? "sunk" : "hit") << endl;
      hits.insert(shot);

      if ((int)hits.size() == k * CARRIER_LENGTH) {
        cerr << "Finished in " << q << " queries." << endl;
        v.check(q <= MAX_QUERIES, "used more than ", MAX_QUERIES, " queries");
        break;
      }
    } else {
      cout << "miss" << endl;
    }
  }

  v.check((int)hits.size() == k * CARRIER_LENGTH, "did not sink all ships within ", 2 * MAX_QUERIES, " queries (aborted)");
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
		argv[0], argv[1], argv[2],
		judge_out,
	};
	return qwerty_getcode(qwerty_main((int)new_argv.size(), new_argv.data()));
}


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
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>

using namespace std;

void check_case() {
  string line;
  /* Get test mode description from judge input file */
  assert(getline(judge_in, line));

  int64_t x, y;

  // Should we make a version of this validator where x
  // and y are actually not fixed until we gave the
  // player enough information, to make sure he's using
  // the best possible guesser, like in guess???

  stringstream sstr(line);
  sstr >> x >> y;

  assert(x <= 1000 * 1000);
  assert(x >= -1000 * 1000);
  assert(y <= 1000 * 1000);
  assert(y >= -1000 * 1000);

  int orientation = 0;
  int64_t playerX = 0, playerY = 0;

  for (int playerV = 1; playerV < 2 * 10 * 1000; playerV++) {
    std::string command;
    if (!std::getline(author_out, command)) {
      wrong_answer("Command %d: couldn't read a command\n", playerV);
      return;
    }

    judge_message("read command %s\n", command.c_str());

    if ((command.length() != 3) || (command[0] != '?') || (command[1] != ' ')) {
      wrong_answer("Wrong command '%s'\n", command.c_str());
      return;
    }
    switch (command[2]) {
    case 'L':
      orientation += 1;
      break;
    case 'R':
      orientation += 3;
      break;
    case 'F':
      break;
    default:
      wrong_answer("Wrong command %s\n", command.c_str());
      return;
    }
    orientation = orientation % 4;
    switch (orientation) {
    case 0:
      playerX += playerV;
      break;
    case 1:
      playerY += playerV;
      break;
    case 2:
      playerX -= playerV;
      break;
    case 3:
      playerY -= playerV;
      break;
    }

    judge_message("player position is now %lld %lld\n", playerX, playerY);

    int64_t distance = llabs(playerX - x) + llabs(playerY - y);

    cout << distance << endl;
    cout.flush();

    if (distance == 0) {
      return;
    }
  }

  wrong_answer("Reached escape velocity");
  return;
}

int main(int argc, char **argv) {
  init_io(argc, argv);

  check_case();

  judge_message("now checking for trailing output");

  /* Check for trailing output. */
  string trash;
  if (author_out >> trash) {
    wrong_answer("Trailing output\n");
  }

  judge_message("accepting");

  /* Yay! */
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

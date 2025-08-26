
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
/*
 Copied from Slack:

 First, note that the optimal solution is always a product of sums (Proof:
 Assume that an optimal solution contains a sub term of the shape a+b*c. Then,
 one of the following three terms is strictly greater, contradicting the
 optimality claim. If b > 1, we can use (a+c)*b instead. If c > 1, we can use
 (a+b)*c. If b = c = 1, we can use a+b+c.). Therefore, we can walk over the AST
 and collect the list of factors. Factorize each factor into primes, and then
 compare the two lists of primes (for the jury answer and for the team output).
 In fact, we don't really need to do the prime factorization for each number,
 we only need to split each 4 into two 2s.

 Reason why we only need to factor the 4s: The only case where we need to form
 a sum that exceeds 4 is when we only have a single 1 and no 2s, in which case
 we must add the 1 to the smallest other number in the input.
*/

#include "validate.h"
#include <cassert>
using namespace OutputValidator;

struct state { // Result of a call to a parsing function.
	Integer value; // Value of the parse.
	bool seenMul; // Has there been a multiplication inside the parsed term?
};

struct parsed {
	Verdict verdict; // The verdict to give when there is an issue. WA for team output, JE for jury output.
	std::string ans; // The string to parse.
	Integer pos; // Current index of parsing.
	std::vector<std::string> parts; // List of substrings that still need to be parsed.
	std::vector<Integer> factors; // List of factors that have been parsed.

	parsed(Verdict verdict_, const std::vector<std::string>& numbers, const std::string& ans_) : verdict(verdict_), ans(ans_), pos(0) {
		// Read all numbers in the string and check that they are the right multiset.
		// As the numbers are treated as strings, this also catches leading zeroes.
		for (Integer i = 0; i < ans.size(); i++) {
			std::string tmp;
			for (; i < ans.size() && isDigit(ans[i]); i++) {
				tmp.push_back(ans[i]);
			}
			if (!tmp.empty()) parts.push_back(tmp);
		}
		std::sort(parts.begin(), parts.end());
		if (parts != numbers) juryOut << "invalid numbers" << verdict;

		// Parse the term and add it as a factor.
		addFactor(parse(true).value);
		// If we did not reach the end of the string while parsing, that's an error.
		if (pos != ans.size()) juryOut << "invalid expression at: " << pos << verdict;

		// Replace every instance of 4 by two instances of 2, then sort.
		for (Integer i = 0; i < factors.size(); i++) {
			if (factors[i] == 4) {
				factors[i] = 2;
				factors.push_back(2);
			}
		}
		std::sort(factors.begin(), factors.end());
	}

	void addFactor(Integer x) {
		if (x > 1) factors.push_back(x);
	}

	// Check that the character at `pos` is `c`, then increment `pos`.
	void consume(char c) {
		if (pos >= ans.size() || ans[pos] != c) juryOut << "invalid expression at: " << pos << verdict;
		pos++;
	}

	// Parse a term consisting of one or more terms separated by `+` or `*`.
	// If `allowMul` is true, we are allowed to have multiplications.
	state parse(bool allowMul) {
		if (pos >= ans.size()) juryOut << "invalid expression at: " << pos << verdict;
		// Parse the leftmost term, record whether it contains a multiplication.
		auto [left, seenMul] = parseSingle(allowMul);
		while (pos < ans.size() && (ans[pos] == '*' || ans[pos] == '+')) {
			// While we only have summations, everything is fine and we can sum up in the variable `left`.
			if (ans[pos] == '+') {
				// There was a multiplication to the left, so this is not optimal.
				if (seenMul) juryOut << "not max a" << verdict;
				consume('+');
				// As we have seen a `+`, don't allow multiplications for future parses.
				allowMul = false;
				// As we don't allow multiplications, we don't need to inspect `seenMul` of the recursive call.
				left += parseSingle(false).value;
			} else {
				// Multiplication after sum, also not optimal.
				if (!allowMul) juryOut << "not max b" << verdict;
				consume('*');
				seenMul = true;
				// We have not seen a `+`, so `left` is just a single term and we add it as a factor.
				addFactor(left);
				// As we know that `allowMul` is true, we can allow multiplications in the recursive call.
				// As we have already seen multiplications, we don't care whether we see more and thus ignore `seenMul`.
				left = parseSingle(true).value;
			}
		}
		if (seenMul) {
			// Add the last remaining number as a factor.
			addFactor(left);
			// Because `seenMul` is true, we can return any value for `left`, it will never be inspected.
			return {0, seenMul};
		} else {
			// No multiplications here, so this sum could be part of a larger sum.
			return {left, seenMul};
		}
	}

	// Parse a single number or a term surrounded by parentheses.
	state parseSingle(bool allowMul) {
		if (pos >= ans.size()) juryOut << "invalid expression at: " << pos << verdict;
		if (ans[pos] == '(') return parsePar(allowMul);
		else if (isDigit(ans[pos])) return parseNumber();
		juryOut << "invalid expression at: " << pos << verdict;
		return {-1, false};
	}

	state parsePar(bool allowMul) {
		consume('(');
		auto res = parse(allowMul);
		consume(')');
		return res;
	}

	// Parse an integer number.
	// Overflow is not an issue, because we already checked that we have the correct multiset of numbers.
	// The same goes for leading zeroes, see the comment above.
	state parseNumber() {
		Integer x = 0;
		while (pos < ans.size() && isDigit(ans[pos])) {
			x *= 10;
			x += ans[pos] - '0';
			pos++;
		}
		return {x, false};
	}

};

int main(int argc, char **argv) {
	OutputValidator::init(argc, argv);

	Integer n = testIn.integer();
	std::vector<std::string> numbers(n);
	for (auto& s : numbers) s = testIn.string();
	sort(numbers.begin(), numbers.end());

	parsed jAns(FAIL, numbers, juryAns.string(1, 1'000'001));
	parsed tAns(WA, numbers, teamAns.string(1, 1'000'001));
	teamAns.newline();
	teamAns.eof();

	if (jAns.factors != tAns.factors) juryOut << "not max" << WA;
	return AC;
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

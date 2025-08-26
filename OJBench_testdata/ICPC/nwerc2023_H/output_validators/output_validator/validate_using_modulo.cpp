#include "validate.h"

#define vl std::vector<Integer>
#define sz(x) (Integer)(x).size()
#define f(i, s, k, l) for (Integer i = s; i < k; i += l)
#define for0(i, k) f(i, 0, k, 1)

std::string parse; // The string to parse.
Integer curIndex; // The current index at which the parsing is.
Integer par = 0; // Current nesting depth of parentheses.

vl parseTerm(Integer open, Verdict v);
// Some large prime numbers.
vl p = {1000000007, 1000000009, 1000000021, 1000000033, 1000000087, 1000000093, 1000000097, 1000000103, 1000000123, 1000000181, 1000000207, 1000000223, 1000000241, 1000000271, 1000000289, 1000000297, 1000000321, 1000000349, 1000000363, 1000000403, 1000000409, 1000000411}; 

using namespace OutputValidator;

bool isOperator(char c) {
    return c == '*' || c == '+'; 
}

void parseError(Verdict v) {
    juryOut << "parsing error" << v; 
}

// Reads one character and advances `curIndex` by 1.
// Also performs a sanity check on the next character after that.
char consume(Verdict v) {
    char c = parse[curIndex]; 
    curIndex++; 
	// Note that we always surround the string to be parsed with parentheses,
	// so that we only need to perform the out-of-bounds check for `curIndex`
	// when `c` is `)`.
    if(isOperator(c)) {
		// Each `+` or `*` must be followed by a digit or an opening parenthesis.
        if(parse[curIndex] != '(' && !isDigit(parse[curIndex])) parseError(v); 
    } else if(c == '(') {
		// Each `(` must be followed by another `(` or a digit.
        if(parse[curIndex] == ')' || isOperator(parse[curIndex])) {       
            parseError(v);
        } 
    } else if(c == ')' ) {
		// Each `)` must be followed by another `)` or an operator.
        if(curIndex < sz(parse) && !isOperator(parse[curIndex]) && parse[curIndex] != ')') parseError(v); 
    } else if(isDigit(c)) {
		// Digits may not be followed by a `(`.
        if(parse[curIndex] == '(') parseError(v); 
    }
	// Any other characters result in an error.
    else parseError(v); 
    return c; 
}

// Parse a positive integer and compute it mod each prime.
vl parseNum(Verdict v) {
    vl res(sz(p), 0);
    while(isDigit(parse[curIndex])) {
        char c = consume(v); 
        for0(i, sz(p)) res[i] = (10 * res[i] + (int) (c - '0')) % p[i]; 
    } 
    return res; 
}

// Parse a term after we just consumed a `*`, multiply the result together with `lval`.
vl parseStar(vl &lval, Integer open, Verdict v) {
    vl rval(sz(p), 0);
    char c = parse[curIndex]; 
    if(isDigit(c)) rval = parseNum(v); 
    else if(c == '(') rval = parseTerm(open, v); 
    for0(i, sz(p)) rval[i] = (lval[i] * rval[i]) % p[i]; 
    return rval; 
}

// Parse a term from the string `parse`, starting at `curIndex`, return value mod each prime.
// `open` is the nesting depth of parentheses at the time the function is called.
vl parseTerm(Integer open, Verdict v) {
    vl num(sz(p), 0); // Number storing the current summand, which might be a product of multiple terms.
    vl s(sz(p), 0); // Sum of all the summands that we parsed so far.

	// Run a loop over all the operator-separated summands.
	// The loop ends when the string ends, or when we get back to the starting nesting depth.
    do {
		// If we are just starting, or the last character we read was a `+`,
		// parse a new number or parenthesized term, and store it into `num`.
        if(isDigit(parse[curIndex])) {
            num = parseNum(v); 
            continue; 
        } 
        char c = consume(v); 
        // cout << c << curIndex << isDigit(c) << endl; 
        if(c == '(') {
            par++; 
            // New sub term, recurse.
            num = parseTerm(open + 1, v); 
        } else if(c == ')') {
            par--; 
            // If we go below the starting depth while parsing, the term is invalid.
            if(par < open) parseError(v); 
        } else if(isDigit(c)) num = parseNum(v); 
		// Found a product, multiply it together with the current value in `num`.
		// Note that here we know that `num` holds some actual number, because
		// of the sanity checks in `consume`.
        else if(c == '*') num = parseStar(num, open, v); 
        else if(c == '+') {
			// Found a sum, so add `num` to the current sum and reset it.
            for0(i, sz(p)) s[i] = (s[i] + num[i]) % p[i]; 
            num = vl(sz(p), 0); 
        }
        else {
			// Any other character results in a parse error.
            parseError(v);
        }
    } while(open != par && curIndex < sz(parse));
	// If the string ended before we got back to the starting depth, it's a parse error.
    if(open != par) parseError(v);  
	// Add the last number to the sum.
    for0(i, sz(p)) s[i] = (s[i] + num[i]) % p[i]; 
    return s;  
}

// Check that the numerical tokens in `ans` are equal to the tokens in `nums`.
void checkNumbers(std::string &ans, std::vector<std::string> &nums, Verdict v) {
    std::vector<std::string> test = {""}; 
    for(char c : ans) {
        if((isOperator(c) || c == '(')) { // At every operator or `(` start a new number.
            if(sz(test.back()) != 0) test.push_back(""); 
        }
        else if(isDigit(c)) test[sz(test) - 1].push_back(c);
        // Note that this behaves weirdly if we get something like `123)456`,
        // but in that case we will hit a parsing error later, so it's fine.
        else if(c == ')') continue; 
        else parseError(v);
    }
    sort(nums.begin(), nums.end()); 
    sort(test.begin(), test.end()); 

    if(sz(test) != sz(nums)) juryOut << "wrong numbers" << WA; 
    for0(i, sz(test)) {
        if(test[i] != nums[i]) juryOut << "wrong numbers" << WA; 
    }
}



int main(int argc, char **argv) {
    OutputValidator::init(argc, argv);                    // initialize streams, rng and parse arguments

    std::string ans = teamAns.string(1, 1'000'001); 
    teamAns.newline();
    teamAns.eof(); 
    std::string correct = juryAns.string(1, 1'000'001); 

    Integer n = testIn.integer();
    std::vector<std::string> a(n);  
    for0(i, n) a[i] = testIn.string();


    parse = '(' + correct + ')'; 
    curIndex = 0; 
    par = 0; 

    // Find the value of the jury expression mod each prime.
    vl jury = parseTerm(0, FAIL); 
    if(curIndex != sz(parse)) parseError(FAIL); 

    // Check that the multiset of numbers is the same as in the input.
    parse = '(' + ans + ')'; 
    checkNumbers(parse, a, WA);
    curIndex = 0; 
    par = 0; 

    // Find the value of the team expression mod each prime.
    vl team = parseTerm(0, WA); 
    if(curIndex != sz(parse)) parseError(WA); 


    // Check if all results are equal. Has a tiny probability to fail.
    for0(i, sz(p)) {
        if(jury[i] != team[i]) juryOut << "not max" << WA; 
    }

    juryOut << AC; 
}

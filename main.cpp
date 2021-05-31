#define DOUBLE_PRECISION 8
#define FAST_IO

#include <bits/stdc++.h>

using namespace std;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

#define ll long long
#define ld long double
#define cld complex<ld>
#define vcld vector<complex<ld>>
#define ull unsigned ll
#define vl vector<ll>
#define vvl vector<vl>
#define vi vector<int>
#define vvi vector<vi>
#define all(x) (x).begin(),(x).end()
#define rall(x) (x).rbegin(),(x).rend()
#define bits(x) __builtin_popcount(x)
#define bit(x, i) ((((x) >> (i)) & 1) != 0)

#ifdef DEBUG
#define dbg(x...) pr("[",#x,"] = ["), db(x), pr("]\n")
#else
#define dbg(x...) 0x12c
#endif

#ifndef DEBUG
#define gcd(x, y) __gcd(x, y)
#endif

const ll INF64 = LLONG_MAX;
const int INF32 = INT_MAX;

void solve();

// @formatter:off
// IN
template<class T> void re(complex<T>& x);
template<class T1, class T2> void re(pair<T1,T2>& p);
template<class T> void re(vector<T>& a);
template<class T, size_t SZ> void re(array<T,SZ>& a);
template<class T> void re(T& x) { cin >> x; }
void re(double& x) { string t; re(t); x = stod(t); }
void re(ld& x) { string t; re(t); x = stold(t); }
template<class T, class... Ts> void re(T& t, Ts&... ts) {re(t); re(ts...);}
template<class T> void re(complex<T>& x) { T a,b; re(a,b); x = cld(a,b); }
template<class T1, class T2> void re(pair<T1,T2>& p) { re(p.first,p.second); }
template<class T> void re(vector<T>& a) { for(auto &v:a) re(v); }
template<class T, size_t SZ> void re(array<T,SZ>& a) { for(auto &v:a) re(v); }
// OUT
void pr(ll x) { cout << x; }
void pr(int x) { pr((ll) x); }
void pr(bool x) { pr(x ? 1: 0); }
void pr(char x) { cout << x; }
void pr(const string &x) { cout << x; }
void pr(const char *x) { cout << x; }
void pr(ld x) { cout << setprecision(DOUBLE_PRECISION) << fixed << x; }
void pr(float x) { pr((ld) x); }
void pr(double x) { pr((ld) x); }
template<class T>
void pr(const T &v);
template<class T, class... Ts>
void pr(T x, Ts... t);
template<class T>
void pr(const complex<T> &x) { pr(x.real(), ' ', x.imag()); }
template<class T1, class T2>
void pr(const pair<T1, T2> &x) { pr(x.first, ' ', x.second); }
template<class T, class... Ts>
void pr(T x, Ts... t) { pr(x), pr(t...); }
template<class T>
void pr(const T &v) {bool fst = true;for (const auto &x: v) {pr(fst ? "" : " ", x), fst = false;}}
void ps() { pr('\n'); }
template<class T, class... Ts>
void ps(const T &x, const Ts &... t) {pr(x);if (sizeof...(t)) { pr(" "); }ps(t...);}
// DEBUG
void db(ll x) { cout << x; }
void db(int x) { db((ll) x); }
void db(char x) { cout << x; }
void db(const string &x) { cout << x; }
void db(const char *x) { cout << x; }
void db(ld x) { cout << setprecision(3) << fixed << x; }
void db(float x) { db((ld) x); }
void db(double x) { db((ld) x); }
template<class T>
void db(const T &v);
template<class T, class... Ts>
void db(T x, Ts... t);
template<class T>
void db(const complex<T> &x) { pr('{', x.real(), ", ", x.imag(), '}'); }
template<class T1, class T2>
void db(const pair<T1, T2> &x) { pr('{', x.first, ", ", x.second, '}'); }
template<class T, class... Ts>
void db(T x, Ts... t) { db(x); if (sizeof...(t)) { pr(", "); } db(t...); }
template<class T>
void db(const T &v) { pr('{'); bool fst = true; for (const auto &x: v) { pr(fst ? "" : ", "); db(x), fst = false; } pr('}'); }
template<class T>
void nvec(const vector<T> &v) {int n = v.size(); ps('{'); for(int i=0;i<n;i++){ps(i," -> ", v[i]);} ps('}');}
// @formatter:on

#define TESTS 0

int main() {
#ifdef FAST_IO
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    std::istream::sync_with_stdio(false);
#endif
#ifdef DEBUG
    auto start = chrono::high_resolution_clock::now();
    freopen("../input.in", "r", stdin);
#endif
    int t = 1;
#if TESTS
    cin >> t;
#endif
    while (t--) {
        solve();
    }
#ifdef DEBUG
    auto end = chrono::high_resolution_clock::now();
    cout << endl << "_______________\nElapsed: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;
#endif
    return 0;
}

//////////////////////// -- YOUR CODE GOES HERE -- ////////////////////////

void solve() {
}
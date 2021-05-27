//
// Created by Andrew on 24.10.2019.
//

// Inversions
template <typename T>
int getSum(T fen[], int index) {
    int sum = 0;
    while (index > 0) {
        sum += fen[index];
        index -= index & (-index);
    }
    return sum;
}

template <typename T>
void updateBIT(T fen[], int n, int index, int val) {
    while (index <= n) {
        fen[index] += val;
        index += index & (-index);
    }
}

template <typename T>
void convert(vector<T> &a) {
    int n = sz(a);
    T temp[n];
    for (int i = 0; i < n; i++)
        temp[i] = a[i];
    sort(temp, temp + n);
    for (int i = 0; i < n; i++) {
        a[i] = lower_bound(temp, temp + n, a[i]) - temp + 1;
    }
}

template <typename T>
int invCount(vector<T> a) {
    int invcount = 0;
    int n = sz(a);
    convert(a);
    T BIT[n + 1];
    for (int i = 1; i <= n; i++)
        BIT[i] = 0;
    for (int i = n - 1; i >= 0; i--) {
        invcount += getSum(BIT, a[i] - 1);
        updateBIT(BIT, n, a[i], 1);
    }
    return invcount;
}

namespace MATRIX {
    const ll mod = 1e9;

    vvl mul(vvl &a, vvl b) {
        assert(sz(a[0]) == sz(b));
        vvl
        res(sz(a), vl(sz(b[0])));
        for (int i = 0; i < sz(a); i++)
            for (int j = 0; j < sz(b[0]); j++)
                for (int k = 0; k < sz(a[0]); k++)
                    res[i][j] = (res[i][j] + (a[i][k] * b[k][j]) % mod) % mod;
        return res;
    }

    vvl sqr(vvl a) {
        return mul(a, a);
    }

    vvl ident(int size) {
        vvl res(size, vl(size));
        for (int i = 0; i < size; i++)
            res[i][i] = 1;
        return res;
    }

    vvl binpow(vvl &a, ll exp) {
        assert(sz(a) == sz(a[0]));
        return (exp == 0 ? ident(sz(a)) : (exp & 1 ? mul(a, binpow(a, exp - 1)) : sqr(binpow(a, exp / 2))));
    }
}

namespace DSU {
    struct dsu {
        map<int, int> p;

        void make(int v) {
            p[v] = v;
        }

        int find(int v) {
            return (v == p[v]) ? v : (p[v] = find(p[v]));
        }

        int merge(int a, int b) {
            a = find(a), b = find(b);
            if (rand() % 2)
                swap(a, b);
            if (a != b)
                p[a] = b;
            return b;
        }
    };
}

namespace TREAP {
    struct node {
        ll x, y;
        int L, R, size = 1;
    };

    vector <node> M(1000000);
    int m_cnt = 1;

    int get_size(int t) {
        return t == 0 ? 0 : M[t].size;
    }

    void recalc(int t) {
        M[t].size = get_size(M[t].L) + get_size(M[t].R) + 1;
    }

    int get_kth(int t, int k) {
        int cur = t;
        while (cur != 0) {
            int s = get_size(M[cur].L);
            if (s == k)
                return cur;
            cur = s > k ? M[cur].L : M[cur].R;
            if (s < k)
                k -= s + 1;
        }
        return 0;
    }

    int create(int x, int y, int L = 0, int R = 0) {
        M[m_cnt] = {x, y, L, R};
        return m_cnt++;
    }

    int merge(int L, int R) {
        if (!L || !R)
            return !R ? L : R;
        if (M[L].y > M[R].y) {
            M[L].R = merge(M[L].R, R);
            recalc(L);
            return L;
        } else {
            M[R].L = merge(L, M[R].L);
            recalc(R);
            return R;
        }
    }

    pair<int, int> split(int T, int key) {
        if (M[T].x <= key) {
            if (!M[T].R)
                return {T, 0};
            else {
                auto t = split(M[T].R, key);
                M[T].R = t.F;
                recalc(T);
                return {T, t.S};
            }
        } else {
            if (!M[T].L)
                return {0, T};
            else {
                auto t = split(M[T].L, key);
                M[T].L = t.S;
                recalc(T);
                return {t.F, T};
            }
        }
    }
}

namespace IMPLTR {
    struct node {
        ll y, w;
        int L, R, size = 1, p = 0;
    };

    vector <node> M(1000000);
    int m_cnt = 1;

    int get_size(int t) {
        return t == 0 ? 0 : M[t].size;
    }

    void recalc(int t) {
        M[t].size = get_size(M[t].L) + get_size(M[t].R) + 1;
        M[M[t].L].p = t;
        M[M[t].R].p = t;
    }

    int get_kth(int t, int k) {
        int cur = t;
        while (cur != 0) {
            int s = get_size(M[cur].L);
            if (s == k)
                return cur;
            cur = s > k ? M[cur].L : M[cur].R;
            if (s < k)
                k -= s + 1;
        }
        return 0;
    }

    int get_ind(int t, int k) {
        int cur = k, res = get_size(M[k].L);
        while (cur != t) {
            if (M[M[cur].p].R == cur)
                res = res + get_size(M[M[cur].p].L) + 1;
            cur = M[cur].p;
        }
        return res;
    }

    int create(int w, int y, int L = 0, int R = 0) {
        M[m_cnt] = {y, w, L, R};
        return m_cnt++;
    }

    int merge(int L, int R) {
        if (!L || !R)
            return !R ? L : R;
        if (M[L].y > M[R].y) {
            M[L].R = merge(M[L].R, R);
            recalc(L);
            return L;
        } else {
            M[R].L = merge(L, M[R].L);
            recalc(R);
            return R;
        }
    }

    pair<int, int> split(int T, int x) {
        int cur_ind = get_size(M[T].L) + 1;
        if (cur_ind <= x) {
            if (!M[T].R)
                return {T, 0};
            else {
                auto t = split(M[T].R, x - cur_ind);
                M[T].R = t.F;
                recalc(T);
                return {T, t.S};
            }
        } else {
            if (!M[T].L)
                return {0, T};
            else {
                auto t = split(M[T].L, x);
                M[T].L = t.S;
                recalc(T);
                return {t.F, T};
            }
        }
    }
}

namespace LARM {
    const ll mod = 1000000000;

    struct verylong {
        vector<int> v;
        int s = 0;

        verylong(ll a = 0) {
            if (a == 0)
                return;
            while (a != 0)
                v.pb(a % mod), a /= mod, s++;
        }

        verylong(vector<int> d) {
            v = d;
            s = v.size();
        }

        int at(int i) const {
            return s == 0 || i > s - 1 ? 0 : v[i];
        }
    };

    ostream &operator<<(ostream &os, const verylong &vv) {
        if (vv.s == 0)
            printf("0");
        else
            for (int i = vv.s - 1; i >= 0; i--)
                if (i == vv.s - 1)
                    printf("%d", vv.at(i));
                else
                    printf("%09d", vv.at(i));
        return os;
    }

    verylong add(verylong a, verylong b) {
//    cout << a << '+' << b << endl;
        vector<int> res(max(a.s, b.s));
        int carry = 0;
        for (int i = 0; i < max(a.s, b.s); i++) {
            res[i] = (a.at(i) + b.at(i) + carry) % mod;
            carry = (a.at(i) + b.at(i) + carry) / mod;
        }
        if (carry)
            res.pb(carry);
        return verylong(res);
    }

    verylong mult(verylong a, ll b) {
        vector<int> res;
        ll carry = 0;
        for (int i = 0; i < a.s; i++) {
            res.pb((a.at(i) * b + carry) % mod);
            carry = (a.at(i) * b + carry) / mod;
        }
        while (carry != 0)
            res.pb(carry % mod), carry /= mod;
        return verylong(res);
    }
}

namespace FFT {
    vector <complex<long double>> fft(const vector <complex<long double>> &as) {
        int n = as.size();
        int k = 0;
        while ((1 << k) < n) k++;
        vector<int> rev(n);
        rev[0] = 0;
        int high1 = -1;
        for (int i = 1; i < n; i++) {
            if ((i & (i - 1)) == 0)
                high1++;
            rev[i] = rev[i ^ (1 << high1)];
            rev[i] |= (1 << (k - high1 - 1));
        }

        vector <complex<long double>> roots(n);
        ld pi = 3.14159265358979323846;
        for (int i = 0; i < n; i++) {
            ld alpha = 2 * pi * i / n;
            roots[i] = complex<long double>(cos(alpha), sin(alpha));
        }

        vector <complex<long double>> cur(n);
        for (int i = 0; i < n; i++)
            cur[i] = as[rev[i]];

        for (int len = 1; len < n; len <<= 1) {
            vector <complex<long double>> ncur(n);
            int rstep = roots.size() / (len * 2);
            for (int pdest = 0; pdest < n;) {
                int p1 = pdest;
                for (int i = 0; i < len; i++) {
                    complex<long double> val = roots[i * rstep] * cur[p1 + len];
                    ncur[pdest] = cur[p1] + val;
                    ncur[pdest + len] = cur[p1] - val;
                    pdest++, p1++;
                }
                pdest += len;
            }
            cur.swap(ncur);
        }
        return cur;
    }

    vector <complex<long double>> fft_rev(const vector <complex<long double>> &as) {
        vector <complex<long double>> res = fft(as);
        for (int i = 0; i < (int) res.size(); i++) res[i] /= as.size();
        reverse(res.begin() + 1, res.end());
        return res;
    }

    vl mult(vl &a, vl &b) {
        assert(a.size() == b.size());
        int n = a.size();
        vcld ca(n * 2), cb(n * 2);
        for (int i = 0; i < n; i++)
            ca[i] = {(ld) a[i], 0}, cb[i] = {(ld) b[i], 0};
        vcld fa = fft(ca), fb = fft(cb);
        vcld res(2 * n);
        for (int i = 0; i < 2 * n; i++)
            res[i] = fa[i] * fb[i];
        res = fft_rev(res);
        vl lres(2 * n);
        for (int i = 0; i < 2 * n; i++)
            lres[i] = (ll) round(res[i].real());
        return lres;
    }

    inline int closest_2(int x) {
        return (1 << (32 - __builtin_clz(x) - (bits(x) == 1)));
    }
}

namespace SEGTREE {
    struct segtree {
        const ll neutral = 0;
        int size = 0;
        vl t, toadd;

        segtree() {
            segtree(0);
        }

        segtree(int size) {
            set_size(size);
        }

        void set_size(int size) {
            this->size = size;
            t.assign(size * 4, neutral);
            toadd.assign(size * 4, 0);
        }

        ll f(ll a, ll b) {
            return a + b;
        }

        void push(int v, int tl, int tr) {
            if (toadd[v] != 0) {
                t[v] += toadd[v] * (tr - tl + 1);
                if (tl != tr) {
                    toadd[2 * v] += toadd[v];
                    toadd[2 * v + 1] += toadd[v];
                }
                toadd[v] = 0;
            }
        }

        void build(vl &a, int v, int tl, int tr) {
            if (tl == tr)
                t[v] = a[tl];
            else {
                int tm = (tl + tr) / 2;
                build(a, 2 * v, tl, tm);
                build(a, 2 * v + 1, tm + 1, tr);
                t[v] = f(t[2 * v], t[2 * v + 1]);
            }
        }

        ll query(int v, int tl, int tr, int l, int r) {
            push(v, tl, tr);
            if (tl > r || tr < l || tl > tr)
                return neutral;
            if (l <= tl && tr <= r)
                return t[v];
            int tm = (tl + tr) / 2;
            ll left = query(2 * v, tl, tm, l, r);
            ll right = query(2 * v + 1, tm + 1, tr, l, r);
            return f(left, right);
        }

        ll query(int l, int r) {
            return query(1, 0, size - 1, l, r);
        }

        void add(int v, int tl, int tr, int l, int r, int val) {
            push(v, tl, tr);
            if (tl > r || tr < l || tl > tr)
                return;
            if (l <= tl && tr <= r)
                toadd[v] = val, push(v, tl, tr);
            else {
                int tm = (tl + tr) / 2;
                add(2 * v, tl, tm, l, r, val);
                add(2 * v + 1, tm + 1, tr, l, r, val);
                t[v] = f(t[2 * v], t[2 * v + 1]);
            }
        }

        void add(int l, int r, int val) {
            add(1, 0, size - 1, l, r, val);
        }

        void print() {
            for (int i = 0; i < size; i++)
                cout << query(i, i) << ' ';
            cout << endl;
        }
    };
}

struct dinic {
    struct edge {
        int a, b;
        ll f = 0, c;

        edge(int a, int b, ll c) : a(a), b(b), c(c) {}
    };

    vector <edge> ve;
    vvi g;
    int n, m = 0;
    int s, t;
    vi lev, ptr;

    dinic(int n) : n(n) {
        g.resize(n);
        lev.resize(n);
        ptr.resize(n);
    }

    void add_edge(int a, int b, ll c1, ll c2) {
        ve.emplace_back(a, b, c1);
        ve.emplace_back(b, a, c2);
        g[a].pb(m);
        g[b].pb(m + 1);
        m += 2;
    }

    bool bfs() {
        fill(all(lev), -1);
        queue<int> q;
        lev[s] = 0;
        q.push(s);
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            for (int id:g[v])
                if (!(ve[id].f >= ve[id].c || lev[ve[id].b] != -1))
                    lev[ve[id].b] = lev[v] + 1, q.push(ve[id].b);
        }
        return lev[t] != -1;
    }

    ll dfs(int v, ll pushed) {
        if (pushed == 0 || v == t)
            return pushed;
        for (; ptr[v] < g[v].size(); ptr[v]++) {
            int id = g[v][ptr[v]];
            int to = ve[id].b;
            if (lev[v] + 1 != lev[to] || ve[id].f >= ve[id].c)
                continue;
            ll tr = dfs(to, min(pushed, ve[id].c - ve[id].f));
            if (tr == 0)
                continue;
            ve[id].f += tr;
            ve[id ^ 1].f -= tr;
            return tr;
        }
        return 0;
    }

    ll flow(int s, int t) {
        this->s = s;
        this->t = t;
        for (auto &e: ve)
            e.f = 0;
        ll f = 0;
        while (bfs()) {
            fill(all(ptr), 0);
            while (ll p = dfs(s, INF64))
                f += p;
        }
        return f;
    }
};

/**
 * Author: Simon Lindholm
 * Date: 2017-04-20
 * License: CC0
 * Source: own work
 * Description: Container where you can add lines of the form kx+m, and query maximum values at points x.
 *  Useful for dynamic programming.
 * Time: O(\log N)
 * Status: tested
 */

//CHT LL

struct Line {
    mutable ll k, m, p;

    bool operator<(const Line &o) const { return k < o.k; }

    bool operator<(ll x) const { return p < x; }
};

struct CHT : multiset<Line, less<>> {
    // (for doubles, use inf = 1/.0, div(a,b) = a/b)
    const ll inf = LLONG_MAX;

    ll div(ll a, ll b) { // floored division
        return a / b - ((a ^ b) < 0 && a % b);
    }

    bool isect(iterator x, iterator y) {
        if (y == end()) {
            x->p = inf;
            return false;
        }
        if (x->k == y->k) x->p = x->m > y->m ? inf : -inf;
        else x->p = div(y->m - x->m, x->k - y->k);
        return x->p >= y->p;
    }

    void add(ll k, ll m) {
        auto z = insert({k, m, 0}), y = z++, x = y;
        while (isect(y, z)) z = erase(z);
        if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
        while ((y = x) != begin() && (--x)->p >= y->p)
            isect(x, erase(y));
    }

    ll query(ll x) {
        assert(!empty());
        auto l = *lower_bound(x);
        return l.k * x + l.m;
    }
};

// CHT Doubles
struct Line {
    ld k, m;
    mutable ld p;

    bool operator<(const Line &o) const {
        return k < o.k;
    }

    bool operator<(const ll &x) const {
        return p < x;
    }
};

struct CHT : multiset<Line, less<>> {
    const ld inf = 1 / .0;

    ld div(ld a, ld b) {
        return a / b;
    }

    bool isect(iterator x, iterator y) {
        if (y == end()) {
            x->p = inf;
            return false;
        }
        if (x->k == y->k) x->p = x->m > y->m ? inf : -inf;
        else x->p = div(y->m - x->m, x->k - y->k);
        return x->p >= y->p;
    }

    void add(ld k, ld m) {
        auto z = insert({k, m, 0}), y = z++, x = y;
        while (isect(y, z)) z = erase(z);
        if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
        while ((y = x) != begin() && (--x)->p >= y->p)
            isect(x, erase(y));
    }

    ld query(ld x) {
        assert(!empty());
        auto l = *lower_bound(x);
        return l.k * x + l.m;
    }
};

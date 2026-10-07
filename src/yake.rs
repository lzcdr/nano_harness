// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! YAKE! — точный порт референсной Python-реализации.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const STOPWORDS_EN_RAW: &str = "dr
dra
mr
ms
a
a's
able
about
above
according
accordingly
across
actually
after
afterwards
again
against
ain't
all
allow
allows
almost
alone
along
already
also
although
always
am
among
amongst
an
and
another
any
anybody
anyhow
anyone
anything
anyway
anyways
anywhere
apart
appear
appreciate
appropriate
are
aren't
around
as
aside
ask
asking
associated
at
available
away
awfully
b
be
became
because
become
becomes
becoming
been
before
beforehand
behind
being
believe
below
beside
besides
best
better
between
beyond
both
brief
but
by
c
c'mon
c's
came
can
can't
cannot
cant
cause
causes
certain
certainly
changes
clearly
co
com
come
comes
concerning
consequently
consider
considering
contain
containing
contains
corresponding
could
couldn't
course
currently
d
definitely
described
despite
did
didn't
different
do
does
doesn't
doing
don't
done
down
downwards
during
e
each
edu
eg
eight
either
else
elsewhere
enough
entirely
especially
et
etc
even
ever
every
everybody
everyone
everything
everywhere
ex
exactly
example
except
f
far
few
fifth
first
five
followed
following
follows
for
former
formerly
forth
four
from
further
furthermore
g
get
gets
getting
given
gives
go
goes
going
gone
got
gotten
greetings
h
had
hadn't
happens
hardly
has
hasn't
have
haven't
having
he
he's
hello
help
hence
her
here
here's
hereafter
hereby
herein
hereupon
hers
herself
hi
him
himself
his
hither
hopefully
how
howbeit
however
i
i'd
i'll
i'm
i've
ie
if
ignored
immediate
in
inasmuch
inc
indeed
indicate
indicated
indicates
inner
insofar
instead
into
inward
is
isn't
it
it'd
it'll
it's
its
itself
j
just
k
keep
keeps
kept
know
knows
known
l
last
lately
later
latter
latterly
least
less
lest
let
let's
like
liked
likely
little
look
looking
looks
ltd
m
mainly
many
may
maybe
me
mean
meanwhile
merely
might
more
moreover
most
mostly
much
must
my
myself
n
name
namely
nd
near
nearly
necessary
need
needs
neither
never
nevertheless
new
next
nine
no
nobody
non
none
noone
nor
normally
not
nothing
novel
now
nowhere
o
obviously
of
off
often
oh
ok
okay
old
on
once
one
ones
only
onto
or
other
others
otherwise
ought
our
ours
ourselves
out
outside
over
overall
own
p
particular
particularly
per
perhaps
placed
please
plus
possible
presumably
probably
provides
q
que
quite
qv
r
rather
rd
re
really
reasonably
regarding
regardless
regards
relatively
respectively
right
s
said
same
saw
say
saying
says
second
secondly
see
seeing
seem
seemed
seeming
seems
seen
self
selves
sensible
sent
serious
seriously
seven
several
shall
she
should
shouldn't
since
six
so
some
somebody
somehow
someone
something
sometime
sometimes
somewhat
somewhere
soon
sorry
specified
specify
specifying
still
sub
such
sup
sure
t
t's
take
taken
tell
tends
th
than
thank
thanks
thanx
that
that's
thats
the
their
theirs
them
themselves
then
thence
there
there's
thereafter
thereby
therefore
therein
theres
thereupon
these
they
they'd
they'll
they're
they've
think
third
this
thorough
thoroughly
those
though
three
through
throughout
thru
thus
to
together
too
took
toward
towards
tried
tries
truly
try
trying
twice
two
u
un
under
unfortunately
unless
unlikely
until
unto
up
upon
us
use
used
useful
uses
using
usually
uucp
v
value
various
very
via
viz
vs
w
want
wants
was
wasn't
way
we
we'd
we'll
we're
we've
welcome
well
went
were
weren't
what
what's
whatever
when
whence
whenever
where
where's
whereafter
whereas
whereby
wherein
whereupon
wherever
whether
which
while
whither
who
who's
whoever
whole
whom
whose
why
will
willing
wish
with
within
without
won't
wonder
would
wouldn't
x
y
yes
yet
you
you'd
you'll
you're
you've
your
yours
yourself
yourselves
z
zero";

const STOPWORDS_RU_RAW: &str = "а
е
и
ж
м
о
на
не
ни
об
но
он
мне
мои
мож
она
они
оно
мной
много
многочисленное
многочисленная
многочисленные
многочисленный
мною
мой
мог
могут
можно
может
можхо
мор
моя
моё
мочь
над
нее
оба
нам
нем
нами
ними
мимо
немного
одной
одного
менее
однажды
однако
меня
нему
меньше
ней
наверху
него
ниже
мало
надо
один
одиннадцать
одиннадцатый
назад
наиболее
недавно
миллионов
недалеко
между
низко
меля
нельзя
нибудь
непрерывно
наконец
никогда
никуда
нас
наш
нет
нею
неё
них
мира
наша
наше
наши
ничего
начала
нередко
несколько
обычно
опять
около
мы
ну
нх
от
отовсюду
особенно
нужно
очень
отсюда
в
во
вон
вниз
внизу
вокруг
вот
восемнадцать
восемнадцатый
восемь
восьмой
вверх
вам
вами
важное
важная
важные
важный
вдали
везде
ведь
вас
ваш
ваша
ваше
ваши
впрочем
весь
вдруг
вы
все
второй
всем
всеми
времени
время
всему
всего
всегда
всех
всею
всю
вся
всё
всюду
г
год
говорил
говорит
года
году
где
да
ее
за
из
ли
же
им
до
по
ими
под
иногда
довольно
именно
долго
позже
более
должно
пожалуйста
значит
иметь
больше
пока
ему
имя
пор
пора
потом
потому
после
почему
почти
посреди
ей
два
две
двенадцать
двенадцатый
двадцать
двадцатый
двух
его
дел
или
без
день
занят
занята
занято
заняты
действительно
давно
девятнадцать
девятнадцатый
девять
девятый
даже
алло
жизнь
далеко
близко
здесь
дальше
для
лет
зато
даром
первый
перед
затем
зачем
лишь
десять
десятый
ею
её
их
бы
еще
при
был
про
процентов
против
просто
бывает
бывь
если
люди
была
были
было
будем
будет
будете
будешь
прекрасно
буду
будь
будто
будут
ещё
пятнадцать
пятнадцатый
друго
другое
другой
другие
другая
других
есть
пять
быть
лучше
пятый
к
ком
конечно
кому
кого
когда
которой
которого
которая
которые
который
которых
кем
каждое
каждая
каждые
каждый
кажется
как
какой
какая
кто
кроме
куда
кругом
с
т
у
я
та
те
уж
со
то
том
снова
тому
совсем
того
тогда
тоже
собой
тобой
собою
тобою
сначала
только
уметь
тот
тою
хорошо
хотеть
хочешь
хоть
хотя
свое
свои
твой
своей
своего
своих
свою
твоя
твоё
раз
уже
сам
там
тем
чем
сама
сами
теми
само
рано
самом
самому
самой
самого
семнадцать
семнадцатый
самим
самими
самих
саму
семь
чему
раньше
сейчас
чего
сегодня
себе
тебе
сеаой
человек
разве
теперь
себя
тебя
седьмой
спасибо
слишком
так
такое
такой
такие
также
такая
сих
тех
чаще
четвертый
через
часто
шестой
шестнадцать
шестнадцатый
шесть
четыре
четырнадцать
четырнадцатый
сколько
сказал
сказала
сказать
ту
ты
три
эта
эти
что
это
чтоб
этом
этому
этой
этого
чтобы
этот
стал
туда
этим
этими
рядом
тринадцать
тринадцатый
этих
третий
тут
эту
суть
чуть
тысяч";

fn stopwords() -> &'static HashSet<String> {
    static SET: OnceLock<HashSet<String>> = OnceLock::new();
    SET.get_or_init(|| {
        let mut s = HashSet::new();
        for line in STOPWORDS_EN_RAW.lines() {
            let t = line.trim();
            if !t.is_empty() {
                s.insert(t.to_string());
            }
        }
        for line in STOPWORDS_RU_RAW.lines() {
            let t = line.trim();
            if !t.is_empty() {
                s.insert(t.to_string());
            }
        }
        s
    })
}

// ==================== Конфиг ====================

#[derive(Debug, Clone)]
pub struct YakeConfig {
    pub top_n: usize,
    pub max_ngram: usize,
    pub window_size: usize,
    pub dedup_lim: f32,
}

impl Default for YakeConfig {
    fn default() -> Self {
        Self {
            top_n: 20,
            max_ngram: 3,
            window_size: 1,
            dedup_lim: 0.9,
        }
    }
}

// ==================== Helpers ====================

/// Python `str.isupper()`: есть хотя бы одна заглавная, и все cased-символы заглавные.
fn py_isupper(s: &str) -> bool {
    let mut has_cased = false;
    for c in s.chars() {
        if c.is_uppercase() {
            has_cased = true;
        } else if c.is_lowercase() {
            return false;
        }
    }
    has_cased
}

fn py_isdigit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// Python `get_tag(word, i, exclude)`.
/// exclude — set(string.punctuation).
fn get_tag(word: &str, i: usize) -> char {
    let no_commas: String = word.chars().filter(|&c| c != ',').collect();
    if py_isdigit(&no_commas) {
        return 'd';
    }
    // убрать первый '.'
    let mut with_one_dot = String::new();
    let mut removed_dot = false;
    for c in no_commas.chars() {
        if c == '.' && !removed_dot {
            removed_dot = true;
            continue;
        }
        with_one_dot.push(c);
    }
    if py_isdigit(&with_one_dot) {
        return 'd';
    }

    let mut cdigit = 0usize;
    let mut calpha = 0usize;
    let mut cexclude = 0usize;
    for c in word.chars() {
        if c.is_ascii_digit() {
            cdigit += 1;
        }
        if c.is_alphabetic() {
            calpha += 1;
        }
        if c.is_ascii_punctuation() {
            cexclude += 1;
        }
    }

    if (cdigit > 0 && calpha > 0) || (cdigit == 0 && calpha == 0) || cexclude > 1 {
        return 'u';
    }

    if py_isupper(word) && !word.is_empty() {
        return 'a';
    }

    let chars: Vec<char> = word.chars().collect();
    if chars.len() > 1 && i > 0 && chars[0].is_uppercase() {
        let upper_count = chars.iter().filter(|c| c.is_uppercase()).count();
        if upper_count == 1 {
            return 'n';
        }
    }

    'p'
}

/// Все символы — пунктуация по ASCII.
fn is_punct_only(w: &str) -> bool {
    !w.is_empty() && w.chars().all(|c| c.is_ascii_punctuation())
}

/// Разбивает предложение на "токены", где слова и пунктуация — отдельные токены.
fn split_sentence_tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_is_word = false;
    let mut started = false;

    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.is_empty() {
            out.push(std::mem::take(cur));
        }
    };

    for c in s.chars() {
        if c.is_whitespace() {
            flush(&mut cur, &mut out);
            started = false;
            continue;
        }
        let is_word = c.is_alphanumeric() || c == '_' || c == '-';
        if !started {
            cur.push(c);
            cur_is_word = is_word;
            started = true;
        } else if is_word == cur_is_word {
            cur.push(c);
        } else {
            flush(&mut cur, &mut out);
            cur.push(c);
            cur_is_word = is_word;
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// Упрощённая замена segtok.split_multi: режем по `. ! ?` + пробел/конец.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if matches!(c, '.' | '!' | '?') {
            let next = chars.peek().copied();
            if next.map(|x| x.is_whitespace()).unwrap_or(true) {
                let t = cur.trim().to_string();
                if !t.is_empty() {
                    out.push(t);
                }
                cur.clear();
            }
        }
    }
    let t = cur.trim().to_string();
    if !t.is_empty() {
        out.push(t);
    }
    out
}

fn median_f32(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

// ==================== Graph ====================

#[derive(Default, Clone)]
struct Graph {
    out: HashMap<usize, HashMap<usize, f32>>,
    inc: HashMap<usize, HashMap<usize, f32>>,
}

impl Graph {
    fn add_node(&mut self, _id: usize) {}

    fn add_edge(&mut self, from: usize, to: usize) {
        *self.out.entry(from).or_default().entry(to).or_insert(0.0) += 1.0;
        *self.inc.entry(to).or_default().entry(from).or_insert(0.0) += 1.0;
    }

    fn has_edge(&self, from: usize, to: usize) -> bool {
        self.out
            .get(&from)
            .map(|m| m.contains_key(&to))
            .unwrap_or(false)
    }

    fn edge_weight(&self, from: usize, to: usize) -> f32 {
        self.out
            .get(&from)
            .and_then(|m| m.get(&to))
            .copied()
            .unwrap_or(0.0)
    }

    /// (wdr, wir, pwr)
    fn out_metrics(&self, id: usize) -> (f32, f32, f32) {
        let empty = HashMap::new();
        let m = self.out.get(&id).unwrap_or(&empty);
        let wdr = m.len() as f32;
        let wir: f32 = m.values().sum();
        let pwr = if wir == 0.0 { 0.0 } else { wdr / wir };
        (wdr, wir, pwr)
    }

    /// (wdl, wil, pwl)
    fn in_metrics(&self, id: usize) -> (f32, f32, f32) {
        let empty = HashMap::new();
        let m = self.inc.get(&id).unwrap_or(&empty);
        let wdl = m.len() as f32;
        let wil: f32 = m.values().sum();
        let pwl = if wil == 0.0 { 0.0 } else { wdl / wil };
        (wdl, wil, pwl)
    }
}

// ==================== Term ====================

#[derive(Default, Clone)]
struct Term {
    stopword: bool,
    tf: f32,
    tf_a: f32,
    tf_n: f32,
    occurs: HashMap<usize, Vec<(usize, usize)>>, // sent_id -> [(pos_sent, pos_text)]
    // features
    h: f32,
    wrel: f32,
    wfreq: f32,
    wspread: f32,
    wcase: f32,
    wpos: f32,
    pl: f32,
    pr: f32,
}

impl Term {
    fn add_occur(&mut self, tag: char, sent_id: usize, pos_sent: usize, pos_text: usize) {
        self.occurs
            .entry(sent_id)
            .or_default()
            .push((pos_sent, pos_text));
        self.tf += 1.0;
        if tag == 'a' {
            self.tf_a += 1.0;
        }
        if tag == 'n' {
            self.tf_n += 1.0;
        }
    }

    /// SingleWord.update_h
    fn update_h(&mut self, stats: &Stats, graph: &Graph, id: usize) {
        let max_tf = stats.max_tf;
        let (wdl, _wil, pwl) = graph.in_metrics(id);
        let (wdr, _wir, pwr) = graph.out_metrics(id);

        self.pl = wdl / max_tf;
        self.pr = wdr / max_tf;
        self.wrel = (0.5 + pwl * (self.tf / max_tf)) + (0.5 + pwr * (self.tf / max_tf));
        self.wfreq = if stats.avg_tf + stats.std_tf > 0.0 {
            self.tf / (stats.avg_tf + stats.std_tf)
        } else {
            0.0
        };
        self.wspread = self.occurs.len() as f32 / stats.n_sentences.max(1) as f32;
        self.wcase = self.tf_a.max(self.tf_n) / (1.0 + self.tf.ln());
        let positions: Vec<f32> = self.occurs.keys().map(|&k| k as f32).collect();
        let med = median_f32(&positions);
        self.wpos = (3.0 + med).ln().ln();

        let denom = self.wcase + self.wfreq / self.wrel + self.wspread / self.wrel;
        self.h = if denom != 0.0 {
            (self.wpos * self.wrel) / denom
        } else {
            0.0
        };
    }
}

struct Stats {
    max_tf: f32,
    avg_tf: f32,
    std_tf: f32,
    n_sentences: usize,
}

// ==================== Candidate ====================

#[derive(Clone)]
struct Candidate {
    kw: String,
    unique_kw: String,
    tags: HashSet<String>,
    term_ids: Vec<usize>,
    tf: f32,
    h: f32,
    start_or_end_stopwords: bool,
}

impl Candidate {
    /// Создаёт кандидата. `terms` — список (tag, word, term_id) в порядке.
    /// Возвращает None если никакие term_ids не заданы (все term_obj отсутствовали — не бывает в нашем случае).
    fn new(terms: &[(char, String, usize)]) -> Self {
        let tag_str: String = terms.iter().map(|(t, _, _)| *t).collect();
        let mut tags = HashSet::new();
        tags.insert(tag_str);

        let kw = terms
            .iter()
            .map(|(_, w, _)| w.clone())
            .collect::<Vec<_>>()
            .join(" ");
        let unique_kw = kw.to_lowercase();
        let term_ids: Vec<usize> = terms.iter().map(|(_, _, i)| *i).collect();

        Self {
            kw,
            unique_kw,
            tags,
            term_ids,
            tf: 0.0,
            h: 1.0,
            start_or_end_stopwords: false, // перезаписывается снаружи через update_start_or_end
        }
    }

    fn update_tags(&mut self, other: &Candidate) {
        for t in &other.tags {
            self.tags.insert(t.clone());
        }
    }

    fn is_valid(&self) -> bool {
        let has_valid = self
            .tags
            .iter()
            .any(|t| !t.contains('u') && !t.contains('d'));
        has_valid && !self.start_or_end_stopwords
    }

    /// ComposedWord.update_h (STOPWORD_WEIGHT="bi")
    fn update_h(&mut self, terms: &[Term], graph: &Graph) {
        let mut sum_h = 0.0f32;
        let mut prod_h = 1.0f32;
        let n = self.term_ids.len();

        for t in 0..n {
            let tid = self.term_ids[t];
            let term_base = &terms[tid];
            if !term_base.stopword {
                sum_h += term_base.h;
                prod_h *= term_base.h;
            } else {
                let mut prob_t1 = 0.0f32;
                if t > 0 {
                    let prev = self.term_ids[t - 1];
                    if graph.has_edge(prev, tid) {
                        prob_t1 = graph.edge_weight(prev, tid) / terms[prev].tf;
                    }
                }
                let mut prob_t2 = 0.0f32;
                if t < n - 1 {
                    let next = self.term_ids[t + 1];
                    if graph.has_edge(tid, next) {
                        prob_t2 = graph.edge_weight(tid, next) / terms[next].tf;
                    }
                }
                let prob = prob_t1 * prob_t2;
                prod_h *= 1.0 + (1.0 - prob);
                sum_h -= 1.0 - prob;
            }
        }

        self.h = prod_h / ((sum_h + 1.0) * self.tf);
    }
}

// ==================== DataCore ====================

struct DataCore {
    terms: Vec<Term>,
    term_by_unique: HashMap<String, usize>,
    candidates: HashMap<String, Candidate>,
    graph: Graph,
    n_sentences: usize,
}

impl DataCore {
    fn new() -> Self {
        Self {
            terms: Vec::new(),
            term_by_unique: HashMap::new(),
            candidates: HashMap::new(),
            graph: Graph::default(),
            n_sentences: 0,
        }
    }

    /// DataCore.get_term
    fn get_term(&mut self, word: &str) -> usize {
        let lower = word.to_lowercase();
        let simples_sto = stopwords().contains(&lower);

        let unique_term = if lower.ends_with('s') && lower.chars().count() > 3 {
            let mut s = lower.clone();
            s.pop();
            s
        } else {
            lower
        };

        if let Some(&id) = self.term_by_unique.get(&unique_term) {
            return id;
        }

        // Убрать всю пунктуацию
        let simples_unique_term: String = unique_term
            .chars()
            .filter(|c| !c.is_ascii_punctuation())
            .collect();

        let isstopword = simples_sto
            || stopwords().contains(&unique_term)
            || simples_unique_term.chars().count() < 3;

        let id = self.terms.len();
        self.terms.push(Term {
            stopword: isstopword,
            ..Default::default()
        });
        self.term_by_unique.insert(unique_term, id);
        self.graph.add_node(id);
        id
    }

    fn add_cooccur(&mut self, left_id: usize, right_id: usize) {
        self.graph.add_edge(left_id, right_id);
    }

    /// DataCore.add_or_update_composedword
    fn add_or_update_composedword(&mut self, mut cand: Candidate) {
        // Вычислить start_or_end_stopwords по первому и последнему term_ids.
        if let (Some(&first), Some(&last)) = (cand.term_ids.first(), cand.term_ids.last()) {
            cand.start_or_end_stopwords = self.terms[first].stopword || self.terms[last].stopword;
        }

        if let Some(existing) = self.candidates.get_mut(&cand.unique_kw) {
            existing.update_tags(&cand);
            existing.tf += 1.0;
        } else {
            cand.tf = 1.0;
            self.candidates.insert(cand.unique_kw.clone(), cand);
        }
    }

    /// DataCore.build_single_terms_features
    fn build_single_terms_features(&mut self) {
        let valid_tfs: Vec<f32> = self
            .terms
            .iter()
            .filter(|t| !t.stopword)
            .map(|t| t.tf)
            .collect();
        if valid_tfs.is_empty() {
            return;
        }
        let n = valid_tfs.len() as f32;
        let avg = valid_tfs.iter().sum::<f32>() / n;
        let var: f32 = valid_tfs.iter().map(|x| (x - avg).powi(2)).sum::<f32>() / n;
        let std = var.sqrt();
        let max_tf = self.terms.iter().map(|t| t.tf).fold(0.0f32, f32::max);

        let stats = Stats {
            max_tf,
            avg_tf: avg,
            std_tf: std,
            n_sentences: self.n_sentences,
        };

        for id in 0..self.terms.len() {
            let mut term = self.terms[id].clone();
            term.update_h(&stats, &self.graph, id);
            self.terms[id] = term;
        }
    }

    /// DataCore.build_mult_terms_features
    fn build_mult_terms_features(&mut self) {
        let terms_snapshot: Vec<Term> = self.terms.clone();
        let graph_snapshot = self.graph.clone();
        for cand in self.candidates.values_mut() {
            if cand.is_valid() {
                cand.update_h(&terms_snapshot, &graph_snapshot);
            }
        }
    }

    /// DataCore._process_word: одна итерация обработки слова.
    /// `block`: уже обработанные слова текущего блока (tag, word, term_id).
    fn process_word(
        &mut self,
        tag: char,
        word: &str,
        sent_id: usize,
        pos_sent: usize,
        pos_text: usize,
        block: &[(char, String, usize)],
        max_ngram: usize,
        window_size: usize,
    ) {
        let term_id = self.get_term(word);
        self.terms[term_id].add_occur(tag, sent_id, pos_sent, pos_text);

        // ко-вхождения
        if tag != 'u' && tag != 'd' {
            let start = if block.len() > window_size {
                block.len() - window_size
            } else {
                0
            };
            for &(prev_tag, _, prev_id) in &block[start..] {
                if prev_tag != 'u' && prev_tag != 'd' {
                    self.add_cooccur(prev_id, term_id);
                }
            }
        }

        // одиночный кандидат
        {
            let terms_vec = vec![(tag, word.to_string(), term_id)];
            let cand = Candidate::new(&terms_vec);
            self.add_or_update_composedword(cand);
        }

        // n-граммы: текущий + до (max_ngram-1) предыдущих из блока
        if max_ngram >= 2 && !block.is_empty() {
            let start = if block.len() > max_ngram - 1 {
                block.len() - (max_ngram - 1)
            } else {
                0
            };
            let mut acc: Vec<(char, String, usize)> = vec![(tag, word.to_string(), term_id)];
            for w in (start..block.len()).rev() {
                acc.push(block[w].clone());
                // acc: [current, prev_last, prev_prev, ...] — разворачиваем
                let mut final_terms = acc.clone();
                final_terms.reverse();
                let cand = Candidate::new(&final_terms);
                self.add_or_update_composedword(cand);
            }
        }
    }
}

// ==================== extract ====================

pub fn extract(text: &str, config: &YakeConfig) -> Vec<(String, f32)> {
    if text.is_empty() {
        return Vec::new();
    }

    // KeywordExtractor: text = text.replace("\n", " ")
    let text = text.replace('\n', " ");
    // pre_filter: split('\n') → 1 часть, добавляем " " и tabs → spaces
    let text = format!(" {}", text.replace('\t', " "));

    let sentences = split_sentences(&text);
    let sentences: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
    if sentences.is_empty() {
        return Vec::new();
    }

    let mut dc = DataCore::new();
    dc.n_sentences = sentences.len();
    let mut pos_text = 0usize;

    for (sent_id, sentence) in sentences.iter().enumerate() {
        let tokens = split_sentence_tokens(sentence);
        let mut block: Vec<(char, String, usize)> = Vec::new();
        let mut pos_sent = 0usize;

        for tok in &tokens {
            if is_punct_only(tok) {
                // граница блока
                if !block.is_empty() {
                    block.clear();
                }
                continue;
            }
            let tag = get_tag(tok, pos_sent);
            dc.process_word(
                tag,
                tok,
                sent_id,
                pos_sent,
                pos_text,
                &block,
                config.max_ngram,
                config.window_size,
            );
            block.push((tag, tok.clone(), dc.term_by_unique[&normalize_word(tok)]));
            pos_sent += 1;
            pos_text += 1;
        }
    }

    dc.build_single_terms_features();
    dc.build_mult_terms_features();

    // Кандидаты: только валидные, сортируем по h (возрастание)
    let mut valid: Vec<Candidate> = dc
        .candidates
        .into_values()
        .filter(|c| c.is_valid())
        .collect();
    valid.sort_by(|a, b| a.h.partial_cmp(&b.h).unwrap_or(std::cmp::Ordering::Equal));

    // Дедупликация
    let mut result: Vec<(String, f32)> = Vec::new();
    for cand in valid {
        let is_dup = result.iter().any(|(kept_kw, _)| {
            let kept_unique = kept_kw.to_lowercase();
            similarity(&cand.unique_kw, &kept_unique) > config.dedup_lim
        });
        if !is_dup {
            result.push((cand.kw, cand.h));
        }
        if result.len() >= config.top_n {
            break;
        }
    }

    result
}

/// Точно совпадает с normalize_word из get_term: lowercase + strip trailing 's' (len > 3).
fn normalize_word(word: &str) -> String {
    let lower = word.to_lowercase();
    if lower.ends_with('s') && lower.chars().count() > 3 {
        let mut s = lower;
        s.pop();
        s
    } else {
        lower
    }
}

// ==================== Similarity ====================

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn similarity(a: &str, b: &str) -> f32 {
    if a == b {
        return 1.0;
    }
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    1.0 - levenshtein(a, b) as f32 / max_len as f32
}

// ==================== Тесты ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text() {
        assert!(extract("", &YakeConfig::default()).is_empty());
    }

    #[test]
    fn top_n_respected() {
        let text = "Rust programming language. Rust is fast. Rust is safe. Memory safety matters.";
        let cfg = YakeConfig {
            top_n: 2,
            ..Default::default()
        };
        let r = extract(text, &cfg);
        assert!(r.len() <= 2);
    }

    #[test]
    fn sorted_ascending() {
        let text = "Rust programming language. Rust is fast and safe. Memory safety matters. Fast performance matters. Compiler is friendly.";
        let cfg = YakeConfig {
            top_n: 10,
            ..Default::default()
        };
        let r = extract(text, &cfg);
        for w in r.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn stopwords_loaded() {
        assert!(stopwords().contains("the"));
        assert!(stopwords().contains("и"));
    }

    #[test]
    fn tag_digit() {
        assert_eq!(get_tag("123", 0), 'd');
        assert_eq!(get_tag("12.5", 0), 'd');
    }

    #[test]
    fn tag_acronym() {
        assert_eq!(get_tag("USA", 0), 'a');
        assert_eq!(get_tag("HTTP", 0), 'a');
    }

    #[test]
    fn tag_proper_noun() {
        assert_eq!(get_tag("Rust", 1), 'n');
        assert_eq!(get_tag("Rust", 0), 'p');
    }

    #[test]
    fn tag_plain() {
        assert_eq!(get_tag("programming", 0), 'p');
    }

    #[test]
    fn punct_only_detects() {
        assert!(is_punct_only(","));
        assert!(is_punct_only("..."));
        assert!(!is_punct_only("a"));
        assert!(!is_punct_only("a,"));
    }

    #[test]
    fn normalize_strips_s() {
        assert_eq!(normalize_word("languages"), "language");
        assert_eq!(normalize_word("was"), "was");
        assert_eq!(normalize_word("Rust"), "rust");
    }

    #[test]
    fn long_english_smoke() {
        let text = "The Rust programming language has come a long way in a few short years, \
from its creation and incubation by a small and nascent community of enthusiasts, \
to becoming one of the most loved and in-demand programming languages in the world. \
Looking back, it was inevitable that the power and promise of Rust would turn heads \
and gain a foothold in systems programming. What was not inevitable was the global \
growth in interest and innovation that permeated through open source communities \
and catalyzed wide-scale adoption across industries.";

        let cfg = YakeConfig {
            top_n: 20,
            ..Default::default()
        };
        let r = extract(text, &cfg);

        eprintln!("YAKE top-20:");
        for (phrase, score) in &r {
            eprintln!("  {:>10.6}  {}", score, phrase);
        }

        assert!(!r.is_empty());
        assert!(r.len() <= 20);
    }
}

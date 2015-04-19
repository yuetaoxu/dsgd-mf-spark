from pyspark import SparkContext
from pyspark import SparkConf

import numpy as np
import time
import sys
import math

FACTORS = 100
WORKERS = 8
ITERATIONS = 50
LAMBDA = 1
INITIAL_LEARNING_RATE = 0.05
BETA = 0.3
NUM_PERM_TRY = 10000
PROFILE = "local-full" # in ["ec2", "local-full", "local-sample"]
TAU0 = None
INPUT_PATH = None
W_PATH = None
H_PATH = None

try:
    FACTORS = int(sys.argv[1])
    WORKERS = int(sys.argv[2])
    ITERATIONS = int(sys.argv[3])
    BETA = float(sys.argv[4])
    LAMBDA = float(sys.argv[5])
    INPUT_PATH = sys.argv[6]
    W_PATH = sys.argv[7]
    H_PATH = sys.argv[8]

    #TAU0 = 100
except:
    pass

USE_FULL_LOSS = False

# set TAU0 so that TAU0 ^ -BETA = INITIAL_LEARNING_RATE
# use a binary search
if TAU0 is None:
    left = 1
    right = 1e100
    for iter in xrange(500):
        mid = (left + right) / 2
        val = mid ** -BETA
        if val < INITIAL_LEARNING_RATE:
            right = mid
        else:
            left = mid
    TAU0 = left
    print >> sys.stderr, "TAU0 automatically set to:", TAU0
    print >> sys.stderr, "Initial learning rate:", TAU0 ** -BETA
else:
    print >> sys.stderr, "TAU0 forced to:", TAU0

# hyper parameters
if PROFILE == "ec2":
    DATA_PATH = "/full/training_set/"
    DATA_FORMAT_IS_TUPLE = False
    STRATA_PATH = "/dat/strata"
elif PROFILE == "local-full":
    DATA_PATH = "dat/full/afs/cs.cmu.edu/project/bigML/netflix-ratings/training_set/mv_*"
    DATA_FORMAT_IS_TUPLE = False
    STRATA_PATH = "dat/strata"
elif PROFILE == "local-sample":
    DATA_PATH = INPUT_PATH if INPUT_PATH is not None else "dat/nf_subsample.csv"
    DATA_FORMAT_IS_TUPLE = True
    STRATA_PATH = "dat/strata"
else:
    print >> sys.stderr, "Invalid setting."
    sys.exit(1)

# time monitor util
class Clock:
    def __init__(self):
        self.time = time.time()

    def toc(self):
        print >> sys.stderr, "Time elapsed:", time.time() - self.time

sc = SparkContext(conf=SparkConf().setAppName("DSGD-MF"))

print >> sys.stderr, "Loading raw files and computing statistics..."

entireTime = Clock()
tic = Clock()

# load data set as tuple (i, j, rating)
if DATA_FORMAT_IS_TUPLE:
    # load from sampled file
    v = sc.textFile(DATA_PATH, use_unicode=False, minPartitions=WORKERS)
    v = v.map(lambda line: tuple([int(x) for x in line.split(",")]))
else:
    # load from full dataset
    def fileToRatingTuples(f):
        lines = f[1].split("\n")
        m = int(lines[0][:-1])
        result = []
        for line in lines[1:-1]:
            firstComma = line.find(",")
            lastComma = line.rfind(",")
            u = int(line[:firstComma])
            r = int(line[firstComma + 1:lastComma])
            result.append((u, m, r))
        return result

    v = sc.wholeTextFiles(DATA_PATH, use_unicode=False, minPartitions=WORKERS)\
        .flatMap(fileToRatingTuples).cache()


# compute basic statistics
def statisticsMapPartition(ratings):
    maxRow = 0
    maxCol = 0
    ni = {}
    nj = {}
    sum = 0
    for x in ratings:
        i, j, r = x
        maxRow = max(i, maxRow)
        maxCol = max(j, maxCol)
        ni[i] = 1 + ni[i] if i in ni else 1
        nj[j] = 1 + nj[j] if j in nj else 1
        sum += r
    result = [("sum", ("add", sum)),
              ("ni", ("aggr", ni)),
              ("nj", ("aggr", nj))]
    return result

def statisticsReduceByKey(x, y):
    if x[0] == "aggr":
        result = {}
        for k, v in x[1].items():
            result[k] = v
        for k, v in y[1].items():
            result[k] = result[k] + v if k in result else v
        return ("aggr", result)
    else:
        return ("sum", x[1] + y[1])

stats = v.mapPartitions(statisticsMapPartition, True).reduceByKeyLocally(statisticsReduceByKey)

ni = stats["ni"][1]
nj = stats["nj"][1]
rows = max(ni.keys()) + 1
cols = max(nj.keys()) + 1
numRatings = sum(ni.values())
avgRating = stats["sum"][1] / float(numRatings)

print >> sys.stderr, "Num rows:", rows
print >> sys.stderr, "Num cols:", cols
print >> sys.stderr, "Num ratings:", numRatings
print >> sys.stderr, "Avg rating:", avgRating

rowsPerBlock = 1 + rows / WORKERS
colsPerBlock = 1 + cols / WORKERS
print >> sys.stderr, "Rows per block:", rowsPerBlock
print >> sys.stderr, "Cols per block:", colsPerBlock

tic.toc()

# generate strata

print >> sys.stderr, "Generating blocks..."
tic = Clock()

def groupPartition(I, xiter):
    blocks = dict([((I, J), []) for J in range(WORKERS)])
    for x in xiter:
        i, j, r = x[0][0], x[0][1], x[1]
        J = j / colsPerBlock
        blocks[(I, J)].append((i << 32) + (j << 16) + r)
    for k in blocks:
        blocks[k] = np.asarray(blocks[k], dtype=np.int64)
    return blocks.items()

vp = v.map(lambda x:((x[0], x[1]), x[2]))\
    .partitionBy(WORKERS, lambda x: x[0] / rowsPerBlock)\
    .mapPartitionsWithIndex(groupPartition, True)
vp.cache()

blockSize = vp.map(lambda x:(x[0], len(x[1]))).collectAsMap()
loss = dict([((I, J), None) for I in range(WORKERS) for J in range(WORKERS)])

v.unpersist()

print >> sys.stderr, "Num blocks generated: %d-by-%d" % (WORKERS, WORKERS)
tic.toc()
tic = Clock()

# broadcast Ni* N*j
nib = [{} for i in xrange(WORKERS)]
njb = [{} for j in xrange(WORKERS)]
for i in ni:
    nib[i / rowsPerBlock][i] = ni[i]
for j in nj:
    njb[j / colsPerBlock][j] = nj[j]
for i in xrange(WORKERS):
    nib[i] = sc.broadcast(nib[i])
    njb[i] = sc.broadcast(njb[i])

# randomize factors so that the expectation of their dot product will be the average rating
randScale = math.sqrt(avgRating * 4 / FACTORS)
w = {}
for i in ni:
    w[i] = np.asarray(np.random.rand(FACTORS), dtype=np.float16) * randScale
h = {}
for j in nj:
    h[j] = np.asarray(np.random.rand(FACTORS), dtype=np.float16) * randScale

# put factors into blocks
wb = [{} for i in xrange(WORKERS)]
hb = [{} for j in xrange(WORKERS)]

for i in w:
    wb[i / rowsPerBlock][i] = w[i]
for j in h:
    hb[j / colsPerBlock][j] = h[j]

del w, h

print >> sys.stderr, "Weight initialization complete."
tic.toc()

# generate a 'good' strata - good means that the blocks are of similar size
def gen_perm_balanced():
    p = list(np.random.permutation(WORKERS))
    bsize = [blockSize[(I, J)] for I, J in enumerate(p)]
    bestGoodness = sum(bsize) / float(max(bsize) * WORKERS)
    bestp = [x for x in p]
    print >> sys.stderr, "Generating strata with initial goodness:", bestGoodness
    for i in range(NUM_PERM_TRY):
        goodness = sum(bsize) / float(max(bsize) * WORKERS)
        if goodness > bestGoodness:
            bestp = [x for x in p]
            bestGoodness = goodness
        else:
            p = bestp
        I, J = np.random.randint(WORKERS, size=2)
        p[I], p[J] = p[J], p[I]
        bsize[I], bsize[J] = blockSize[(I, p[I])], blockSize[(J, p[J])]
    print >> sys.stderr, "Optimized strata with goodness %f: %s" % (bestGoodness, bestp)
    return bestp

def gen_perm_naive():
    return np.random.permutation(WORKERS)

n = 0
for iteration in range(1, ITERATIONS + 1):

    print >> sys.stderr, "Iteration", iteration
    approxLR = (TAU0 + n) ** (-BETA)
    #n0 = TAU0 + n
    print >> sys.stderr, "Approx learning rate:", approxLR
    tic = Clock()

    # generate permutation and broadcast W & H
    perm = gen_perm_naive()
    print >> sys.stderr, "Broadcasting...",
    btic = Clock()
    wbb = [sc.broadcast(x) for x in wb]
    hbb = [sc.broadcast(x) for x in hb]

    permBc = sc.broadcast(perm)
    btic.toc()

    # training function (as mapPartition)
    def train(block):
        I, J = block[0][0], block[0][1]
        viter = block[1]
        NIB = nib[I].value
        NJB = njb[J].value

        WB = wbb[I].value
        HB = hbb[J].value

        for i in WB:
            WB[i] = np.asarray(WB[i], dtype=np.float32)
        for j in HB:
            HB[j] = np.asarray(HB[j], dtype=np.float32)

        blockLoss = 0

        for k, v in enumerate(viter):
            i, v = v >> 32, v & 0xffffffff
            j, r = v >> 16, v & 0xffff
            delta = WB[i].dot(HB[j]) - r
            blockLoss += delta * delta
            #delta = min(1, delta)
            #delta = max(-1, delta)
            WB[i] -= approxLR * (2.0 * delta * HB[j] + (2.0 * LAMBDA / NIB[i]) * WB[i])
            HB[j] -= approxLR * (2.0 * delta * WB[i] + (2.0 * LAMBDA / NJB[j]) * HB[j])

        for i in WB:
            WB[i] = np.asarray(WB[i], dtype=np.float16)
        for j in HB:
            HB[j] = np.asarray(HB[j], dtype=np.float16)

        result = [("l", I, J, blockLoss), ("w", I, WB), ("h", J, HB)]
        return result

    update = vp.filter(lambda x:x[0][1] == permBc.value[x[0][0]])\
        .flatMap(train, True).collect()

    # update the parameters
    for u in update:
        if u[0] == "w":
            wb[u[1]] = u[2]
        elif u[0] == "h":
            hb[u[1]] = u[2]
        elif u[0] == "l":
            loss[(u[1], u[2])] = u[3]

    # update the n (to update the learning rate)
    for I, J in enumerate(perm):
        n += blockSize[(I, J)]
    print >> sys.stderr, "Num updates:", n

    # estimate loss and RMSE
    if not USE_FULL_LOSS:
        estLoss = 0
        estRatings = 0
        for I in range(WORKERS):
            for J in range(WORKERS):
                if loss[(I, J)] is not None:
                    estLoss += loss[(I, J)]
                    estRatings += blockSize[(I, J)]
        estLoss = estLoss * float(numRatings) / estRatings
        estError = math.sqrt(estLoss / numRatings)
        print >> sys.stderr, "Estimated loss:", estLoss
        print >> sys.stderr, "Estimated RMSE:", estError
    else:
        # slow & mem-intensive exact LOSS computation
        # only feasible for small dataset
        exactLoss = 0
        for i, j, r in v.collect():
            error = wb[i / rowsPerBlock][i].dot(hb[j / colsPerBlock][j]) - r
            exactLoss += error * error
        error = math.sqrt(exactLoss / numRatings)
        print >> sys.stderr, "Exact loss:", exactLoss
        print >> sys.stderr, "Exact RMSE:", error

    tic.toc()

# save W and H
if W_PATH is not None:
    wf = open(W_PATH, "w")
    for i in range(1, rows):
        try:
            wi = wb[i / rowsPerBlock][i]
        except:
            wi = np.zeros(FACTORS)
        print >> wf, ",".join(map(lambda x:str(x), wi))
    wf.close()
    print >> sys.stderr, "W saved to", W_PATH

if H_PATH is not None:
    hf = open(H_PATH, "w")
    h = []
    for j in range(cols):
        try:
            h.append(hb[j / colsPerBlock][j])
        except:
            h.append(np.zeros(FACTORS))
    for i in range(FACTORS):
        print >> hf, ",".join([str(h[j][i]) for j in range(1, cols)])
    hf.close()
    print >> sys.stderr, "H saved to", H_PATH

entireTime.toc()

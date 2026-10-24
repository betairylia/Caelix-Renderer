// The reusable GPU hash table (docs/HASH_TABLE.md; lib/src/rhi/hash_table.h) against a CPU
// model, without the rest of the library: the test owns the buffers (host-visible, so it reads every
// table word), runs the library's maintenance kernels from the build's shader folder in the order of
// HashTable::RecordMaintenance, and drives the device API (common/caelix_hash_table.hlsl) with the
// probe kernel tests/shaders/caelix_hash_probe.hlsl from many threads at once:
//   - the fingerprint against the CPU model;
//   - contention: one key from 65536 threads (exactly one insert, no duplicate slot), many keys with
//     duplicates from 100k threads (per key at most one insert, every hit on the inserter's payload,
//     the payload headers, no duplicate key or payload in the table);
//   - collisions: 40 keys with one home slot (8 inserted, the rest a full window), two keys with the
//     same key_lo (pending in the first frame, both present after it);
//   - payload exhaustion: inserts beyond the free stack, entries without payload, the stack fix-up,
//     recovery after eviction;
//   - eviction at the max_age boundary, the key rebuild (every live payload found again), the active
//     list and the indirect dispatch of the user's pass, the counters;
//   - the free stack and the live payloads stay a partition of the pool over many random frames.
#include "test_kernels.h"
#include "test_scenes.h"

#include "rhi/hash_table_layout.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

#ifndef CAELIX_TEST_SHADER_DIR
#error "CAELIX_TEST_SHADER_DIR must name the build's shader folder"
#endif

namespace {

using caelix::HashCounters;
using caelix::HashKeySlot;
using caelix::HashPush;

using Words = std::array<uint32_t, caelix::kHashMaxKeyWords>;

// ---------------------------------------------------------------------------
// CPU model of HashFingerprint

uint64_t SplitMix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

std::array<uint32_t, 2> Fingerprint(const Words& words, uint32_t count)
{
    uint64_t h = 0;
    for (uint32_t i = 0; i < count; i += 2)
    {
        const uint64_t high = i + 1 < count ? words[i + 1] : 0u;
        h = SplitMix64(h ^ ((high << 32) | words[i]));
    }
    const uint32_t lo = static_cast<uint32_t>(h);
    const uint32_t hi = static_cast<uint32_t>(h >> 32);
    return {std::max(lo, 1u), hi != 0 ? hi : 1u};
}

// ---------------------------------------------------------------------------
// The table's buffers, the library's maintenance kernels and the probe

// Push constants of tests/shaders/caelix_hash_probe.hlsl (the same member names).
struct HashProbePush
{
    uint64_t keys;
    uint64_t stack;
    uint64_t counters;
    uint64_t payloads;
    uint64_t input;
    uint64_t output;
    uint64_t extra;
    uint32_t key_mask;
    uint32_t payload_stride;
    uint32_t key_words;
    uint32_t frame;
    uint32_t mode;
    uint32_t count;
};
static_assert(sizeof(HashProbePush) == 80);

constexpr uint32_t kModeFingerprint = 0;
constexpr uint32_t kModeInsert = 1;
constexpr uint32_t kModeLookup = 2;
constexpr uint32_t kModeUserPass = 3;
constexpr uint32_t kItemsPerGroup = 64;  // the probe's user pass: 64 payloads per workgroup

struct Result
{
    uint32_t result;
    uint32_t payload;
};

class Table
{
public:
    Table(test::Gpu& gpu, uint32_t capacity, uint32_t key_words, uint32_t max_inputs)
        : gpu_(gpu),
          capacity_(capacity),
          key_words_(key_words),
          stride_(caelix::kHashPayloadKey + 4 * key_words),
          slots_(std::bit_ceil(capacity * caelix::kHashKeySlotsPerPayload)),
          init_(gpu, CAELIX_TEST_SHADER_DIR, "caelix_hash_init", sizeof(HashPush)),
          begin_(gpu, CAELIX_TEST_SHADER_DIR, "caelix_hash_begin", sizeof(HashPush)),
          maintain_(gpu, CAELIX_TEST_SHADER_DIR, "caelix_hash_maintain", sizeof(HashPush)),
          finish_(gpu, CAELIX_TEST_SHADER_DIR, "caelix_hash_finish", sizeof(HashPush)),
          probe_(gpu, CAELIX_TEST_SHADER_DIR, "caelix_hash_probe", sizeof(HashProbePush))
    {
        keys_ = test::CreateHostBuffer(gpu, uint64_t{slots_} * caelix::kHashSlotBytes);
        stack_ = test::CreateHostBuffer(gpu, caelix::HashStackBytes(capacity));
        counters_ = test::CreateHostBuffer(gpu, caelix::kHashCounterBytes);
        payloads_ = test::CreateHostBuffer(gpu, uint64_t{capacity} * stride_);
        active_ = test::CreateHostBuffer(gpu, uint64_t{capacity} * 4);
        dispatch_ = test::CreateHostBuffer(gpu, caelix::kHashDispatchBytes);
        input_ = test::CreateHostBuffer(gpu, uint64_t{max_inputs} * 32);
        const uint64_t output_bytes = std::max<uint64_t>(uint64_t{max_inputs} * 8, uint64_t{capacity} * 4);
        output_ = test::CreateHostBuffer(gpu, output_bytes);
    }
    ~Table()
    {
        for (test::HostBuffer* b :
             {&keys_, &stack_, &counters_, &payloads_, &active_, &dispatch_, &input_, &output_})
        {
            test::DestroyHostBuffer(gpu_, *b);
        }
    }

    HashPush Push(uint32_t frame, uint32_t max_age) const
    {
        HashPush push{};
        push.keys = keys_.address;
        push.stack = stack_.address;
        push.counters = counters_.address;
        push.payloads = payloads_.address;
        push.active = active_.address;
        push.dispatch = dispatch_.address;
        push.key_mask = slots_ - 1;
        push.payload_capacity = capacity_;
        push.payload_stride = stride_;
        push.key_words = key_words_;
        push.frame = frame;
        push.max_age = max_age;
        push.items_per_group = kItemsPerGroup;
        return push;
    }

    // The maintenance of `frame`, in the order of HashTable::RecordMaintenance (the first call
    // initialises the pool), then the probe's user pass over the active list (writes `frame` to
    // output[payload], marks the payload ready). Returns the counters of the pass.
    HashCounters Maintain(uint32_t frame, uint32_t max_age)
    {
        const HashPush push = Push(frame, max_age);
        const uint32_t groups = (capacity_ + caelix::kHashThreads - 1) / caelix::kHashThreads;
        std::fill(output_.data, output_.data + output_.size, uint8_t{0});
        test::SubmitAndWait(gpu_, [&](VkCommandBuffer cmd) {
            if (!initialised_)
            {
                init_.Dispatch(cmd, &push, groups);
                initialised_ = true;
            }
            begin_.Record(cmd, &push, 1);
            vkCmdFillBuffer(cmd, keys_.buffer, 0, VK_WHOLE_SIZE, 0);
            test::ComputeBarrier(cmd);
            maintain_.Dispatch(cmd, &push, groups);
            finish_.Dispatch(cmd, &push, 1);
            HashProbePush probe = ProbePush(frame, kModeUserPass, 0);
            probe.input = active_.address;
            probe.extra = dispatch_.address;
            probe_.RecordIndirect(cmd, &probe, dispatch_.buffer, 0);
            test::ComputeBarrier(cmd);
        });
        return Counters();
    }

    // One probe dispatch over `inputs` (mode insert or lookup) at `frame`.
    std::vector<Result> Run(uint32_t mode, uint32_t frame, const std::vector<Words>& inputs)
    {
        REQUIRE(inputs.size() * 32 <= input_.size);
        std::memcpy(input_.data, inputs.data(), inputs.size() * sizeof(Words));
        const HashProbePush push = ProbePush(frame, mode, static_cast<uint32_t>(inputs.size()));
        test::SubmitAndWait(gpu_, [&](VkCommandBuffer cmd) {
            probe_.Dispatch(cmd, &push, static_cast<uint32_t>((inputs.size() + 63) / 64));
        });
        std::vector<Result> out(inputs.size());
        if (mode == kModeFingerprint)
        {
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                out[i] = {output_.As<uint32_t>()[2 * i], output_.As<uint32_t>()[2 * i + 1]};
            }
            return out;
        }
        std::memcpy(out.data(), output_.data, out.size() * sizeof(Result));
        return out;
    }

    HashCounters Counters() const
    {
        HashCounters c{};
        std::memcpy(&c, counters_.data, sizeof(c));
        return c;
    }
    int32_t Top() const
    {
        return stack_.As<int32_t>()[0];
    }
    uint32_t StackEntry(uint32_t i) const
    {
        return stack_.As<uint32_t>()[1 + i];
    }
    const HashKeySlot& Slot(uint32_t i) const
    {
        return keys_.As<HashKeySlot>()[i];
    }
    const uint32_t* Payload(uint32_t i) const
    {
        return reinterpret_cast<const uint32_t*>(payloads_.data + uint64_t{i} * stride_);
    }
    uint32_t PassFrame(uint32_t payload) const
    {
        return output_.As<uint32_t>()[payload];
    }
    const caelix::HashDispatch& Dispatch() const
    {
        return *dispatch_.As<caelix::HashDispatch>();
    }
    uint32_t ActiveEntry(uint32_t i) const
    {
        return active_.As<uint32_t>()[i];
    }
    uint32_t slots() const
    {
        return slots_;
    }
    uint32_t capacity() const
    {
        return capacity_;
    }
    uint32_t key_words() const
    {
        return key_words_;
    }

    // Every published key of the table (key_lo, key_hi) -> its slots (a key must have one).
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> Keys() const
    {
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> keys;
        for (uint32_t s = 0; s < slots_; ++s)
        {
            const HashKeySlot& slot = Slot(s);
            if (slot.key_lo != 0)
            {
                keys[{slot.key_lo, slot.key_hi}].push_back(s);
            }
        }
        return keys;
    }

private:
    HashProbePush ProbePush(uint32_t frame, uint32_t mode, uint32_t count) const
    {
        HashProbePush push{};
        push.keys = keys_.address;
        push.stack = stack_.address;
        push.counters = counters_.address;
        push.payloads = payloads_.address;
        push.input = input_.address;
        push.output = output_.address;
        push.key_mask = slots_ - 1;
        push.payload_stride = stride_;
        push.key_words = key_words_;
        push.frame = frame;
        push.mode = mode;
        push.count = count;
        return push;
    }

    test::Gpu& gpu_;
    uint32_t capacity_;
    uint32_t key_words_;
    uint32_t stride_;
    uint32_t slots_;
    bool initialised_ = false;
    test::ComputeKernel init_;
    test::ComputeKernel begin_;
    test::ComputeKernel maintain_;
    test::ComputeKernel finish_;
    test::ComputeKernel probe_;
    test::HostBuffer keys_;
    test::HostBuffer stack_;
    test::HostBuffer counters_;
    test::HostBuffer payloads_;
    test::HostBuffer active_;
    test::HostBuffer dispatch_;
    test::HostBuffer input_;
    test::HostBuffer output_;
};

Words RandomWords(std::mt19937& rng, uint32_t count)
{
    Words w{};
    for (uint32_t i = 0; i < count; ++i)
    {
        w[i] = rng();
    }
    return w;
}

// Checks of every insert dispatch: per key at most one kHashInserted, every hit on the payload of the
// key's slot, no key twice in the table, no payload twice, every payload header holds its key.
// Returns the payload of every key that has one (key index -> payload).
std::map<size_t, uint32_t> CheckInserts(const Table& t, const std::vector<Words>& keys,
                                        const std::vector<size_t>& key_of_thread,
                                        const std::vector<Result>& results)
{
    std::vector<uint32_t> inserted(keys.size(), 0);
    std::map<size_t, uint32_t> payload_of;
    for (size_t i = 0; i < results.size(); ++i)
    {
        const Result& r = results[i];
        REQUIRE(r.result <= caelix::kHashEmptyEntry);
        if (r.result == caelix::kHashInserted || r.result == caelix::kHashHit)
        {
            REQUIRE(r.payload < t.capacity());
            const auto [it, added] = payload_of.emplace(key_of_thread[i], r.payload);
            REQUIRE(it->second == r.payload);
            static_cast<void>(added);
        }
        if (r.result == caelix::kHashInserted)
        {
            ++inserted[key_of_thread[i]];
        }
    }
    for (const uint32_t n : inserted)
    {
        REQUIRE(n <= 1);
    }
    // The table: every key once; its payload (or none) is the one the threads saw.
    const auto table = t.Keys();
    std::set<uint32_t> payloads;
    std::map<std::pair<uint32_t, uint32_t>, size_t> key_index;
    for (size_t k = 0; k < keys.size(); ++k)
    {
        const auto fp = Fingerprint(keys[k], t.key_words());
        key_index[{fp[0], fp[1]}] = k;
    }
    for (const auto& [key, slots] : table)
    {
        REQUIRE(slots.size() == 1);
        const HashKeySlot& slot = t.Slot(slots[0]);
        REQUIRE(key.second != 0);  // nothing left pending after the dispatch
        // The slot lies in the key's window.
        REQUIRE(((slots[0] - key.first) & (t.slots() - 1)) < caelix::kHashProbes);
        if (slot.payload == caelix::kHashNone)
        {
            continue;
        }
        REQUIRE(payloads.insert(slot.payload).second);
        const uint32_t* header = t.Payload(slot.payload);
        REQUIRE((header[0] == caelix::kHashStateUninitialised || header[0] == caelix::kHashStateReady));
        // The payload's header holds this key (keys of earlier frames included).
        Words w{};
        for (uint32_t i = 0; i < t.key_words(); ++i)
        {
            w[i] = header[2 + i];
        }
        const auto fp = Fingerprint(w, t.key_words());
        REQUIRE(fp[0] == key.first);
        REQUIRE(fp[1] == key.second);
        const auto k = key_index.find(key);
        if (k != key_index.end())
        {
            const auto seen = payload_of.find(k->second);
            if (seen != payload_of.end())
            {
                REQUIRE(seen->second == slot.payload);
            }
        }
    }
    // Every key a thread inserted or hit is in the table.
    for (const auto& [k, payload] : payload_of)
    {
        const auto fp = Fingerprint(keys[k], t.key_words());
        const auto found = table.find({fp[0], fp[1]});
        REQUIRE(found != table.end());
        REQUIRE(t.Slot(found->second[0]).payload == payload);
    }
    return payload_of;
}

// The free stack entries [0, top) and the payloads whose state is not free are a partition of the
// pool.
void CheckPartition(const Table& t)
{
    const int32_t top = t.Top();
    REQUIRE(top >= 0);
    std::vector<int> seen(t.capacity(), 0);
    for (int32_t i = 0; i < top; ++i)
    {
        const uint32_t e = t.StackEntry(static_cast<uint32_t>(i));
        REQUIRE(e < t.capacity());
        REQUIRE(t.Payload(e)[0] == caelix::kHashStateFree);
        ++seen[e];
    }
    for (uint32_t p = 0; p < t.capacity(); ++p)
    {
        if (t.Payload(p)[0] != caelix::kHashStateFree)
        {
            ++seen[p];
        }
        INFO("payload " << p);
        REQUIRE(seen[p] == 1);
    }
}

// After a maintenance: every payload that is not free has its key in its window (with its payload),
// the active list holds exactly the payloads touched at frame - 1 (and the user pass saw them), the
// dispatch covers them, and live / active count them.
void CheckRebuilt(const Table& t, uint32_t frame, const HashCounters& c)
{
    const auto table = t.Keys();
    uint32_t live = 0;
    uint32_t active = 0;
    std::set<uint32_t> active_set;
    for (uint32_t i = 0; i < c.active; ++i)
    {
        REQUIRE(active_set.insert(t.ActiveEntry(i)).second);
    }
    for (uint32_t p = 0; p < t.capacity(); ++p)
    {
        const uint32_t* header = t.Payload(p);
        if (header[0] == caelix::kHashStateFree)
        {
            REQUIRE(active_set.count(p) == 0);
            continue;
        }
        ++live;
        Words w{};
        for (uint32_t k = 0; k < t.key_words(); ++k)
        {
            w[k] = header[2 + k];
        }
        const auto fp = Fingerprint(w, t.key_words());
        const auto found = table.find({fp[0], fp[1]});
        REQUIRE(found != table.end());
        REQUIRE(found->second.size() == 1);
        REQUIRE(t.Slot(found->second[0]).payload == p);
        const bool is_active = header[1] == frame - 1;
        REQUIRE(active_set.count(p) == (is_active ? 1u : 0u));
        REQUIRE(t.PassFrame(p) == (is_active ? frame : 0u));
        active += is_active ? 1u : 0u;
    }
    REQUIRE(table.size() == live);
    REQUIRE(c.live == live);
    REQUIRE(c.active == active);
    const caelix::HashDispatch& d = t.Dispatch();
    const uint32_t groups = (active + kItemsPerGroup - 1) / kItemsPerGroup;
    REQUIRE(d.count == active);
    REQUIRE(d.z == 1);
    REQUIRE(uint64_t{d.x} * d.y >= groups);
    REQUIRE(d.x <= caelix::kHashDispatchRowGroups);
}

}  // namespace

TEST_CASE("hash table: the fingerprint matches the CPU model", "[hash_table]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_hash_table_test", test::RigDesc(0, 0, 0));
    {
        std::mt19937 rng(11);
        for (uint32_t words = 1; words <= caelix::kHashMaxKeyWords; ++words)
        {
            Table t(rig.gpu, 64, words, 4096);
            std::vector<Words> inputs(4096);
            for (Words& w : inputs)
            {
                w = RandomWords(rng, words);
            }
            inputs[0] = Words{};  // all zero words: the forced key_lo / key_hi rules
            const std::vector<Result> out = t.Run(kModeFingerprint, 2, inputs);
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                const auto fp = Fingerprint(inputs[i], words);
                INFO("words " << words << " input " << i);
                REQUIRE(out[i].result == fp[0]);
                REQUIRE(out[i].payload == fp[1]);
                REQUIRE(fp[0] >= 1);
                REQUIRE(fp[1] != 0);
            }
        }
    }
    test::CloseRig(rig);
}

TEST_CASE("hash table: contention on one key and on many keys", "[hash_table]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_hash_table_test", test::RigDesc(0, 0, 0));
    {
        std::mt19937 rng(21);
        // 1. One key from 65536 threads: one insert, the others pending or a hit on its payload.
        {
            Table t(rig.gpu, 1024, 3, 65536);
            const HashCounters first = t.Maintain(2, 128);
            REQUIRE(first.live == 0);
            REQUIRE(t.Top() == 1024);
            const std::vector<Words> keys = {RandomWords(rng, 3)};
            const std::vector<Words> inputs(65536, keys[0]);
            const std::vector<size_t> key_of(inputs.size(), 0);
            const std::vector<Result> results = t.Run(kModeInsert, 2, inputs);
            const auto payload_of = CheckInserts(t, keys, key_of, results);
            REQUIRE(payload_of.size() == 1);
            size_t inserted = 0;
            size_t pending = 0;
            for (const Result& r : results)
            {
                inserted += r.result == caelix::kHashInserted ? 1 : 0;
                pending += r.result == caelix::kHashPending ? 1 : 0;
                REQUIRE((r.result == caelix::kHashInserted || r.result == caelix::kHashPending ||
                         r.result == caelix::kHashHit));
            }
            REQUIRE(inserted == 1);
            const HashCounters c = t.Counters();
            REQUIRE(c.inserted == 1);
            REQUIRE(c.pending == pending);
            REQUIRE(t.Top() == 1023);
            std::printf("one key from 65536 threads: 1 insert, %zu pending, %zu hits\n", pending,
                        results.size() - pending - 1);
            // The same frame again: every thread hits.
            for (const Result& r : t.Run(kModeLookup, 2, inputs))
            {
                REQUIRE(r.result == caelix::kHashHit);
                REQUIRE(r.payload == payload_of.begin()->second);
            }
            // After the maintenance of the next frame: rebuilt, active, and still a hit.
            const HashCounters m = t.Maintain(3, 128);
            CheckRebuilt(t, 3, m);
            REQUIRE(m.live == 1);
            REQUIRE(m.active == 1);
            for (const Result& r : t.Run(kModeLookup, 3, {keys[0]}))
            {
                REQUIRE(r.result == caelix::kHashHit);
                REQUIRE(r.payload == payload_of.begin()->second);
            }
            CheckPartition(t);
        }
        // 2. 3000 keys, 100k threads in random order (about 33 threads per key), over two frames.
        {
            Table t(rig.gpu, 4096, 7, 100000);
            t.Maintain(2, 128);
            std::vector<Words> keys(3000);
            for (Words& k : keys)
            {
                k = RandomWords(rng, 7);
            }
            std::vector<Words> inputs(100000);
            std::vector<size_t> key_of(inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                key_of[i] = rng() % keys.size();
                inputs[i] = keys[key_of[i]];
            }
            const std::vector<Result> results = t.Run(kModeInsert, 2, inputs);
            const auto payload_of = CheckInserts(t, keys, key_of, results);
            const HashCounters c = t.Counters();
            std::printf("3000 keys from 100000 threads: %u inserted, %u pending, %u full windows\n",
                        c.inserted, c.pending, c.failed_slot);
            REQUIRE(c.inserted == payload_of.size());
            REQUIRE(c.failed_payload == 0);
            REQUIRE(t.Top() == static_cast<int32_t>(4096 - c.inserted));
            // Fill 3000 / 16384 = 0.18: a full window is rare.
            REQUIRE(c.inserted >= 2990);
            const HashCounters m = t.Maintain(3, 128);
            CheckRebuilt(t, 3, m);
            REQUIRE(m.live == c.inserted);
            REQUIRE(m.active == c.inserted);
            // Every inserted key is a hit on its payload; a lookup touches it.
            std::vector<Words> all(keys.begin(), keys.end());
            const std::vector<Result> again = t.Run(kModeLookup, 3, all);
            for (size_t k = 0; k < keys.size(); ++k)
            {
                const auto p = payload_of.find(k);
                if (p != payload_of.end())
                {
                    REQUIRE(again[k].result == caelix::kHashHit);
                    REQUIRE(again[k].payload == p->second);
                    REQUIRE(t.Payload(p->second)[1] == 3u);
                }
                else
                {
                    REQUIRE(again[k].result == caelix::kHashMissing);
                }
            }
            CheckPartition(t);
        }
    }
    test::CloseRig(rig);
}

TEST_CASE("hash table: colliding home slots and equal key_lo", "[hash_table]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_hash_table_test", test::RigDesc(0, 0, 0));
    {
        std::mt19937 rng(31);
        // 1. 40 keys with the same home slot, each from 64 threads: the window takes 8 of them.
        {
            Table t(rig.gpu, 1024, 2, 4096);
            t.Maintain(2, 128);
            const uint32_t mask = t.slots() - 1;
            std::vector<Words> keys;
            uint32_t home = UINT32_MAX;
            while (keys.size() < 40)
            {
                const Words w = RandomWords(rng, 2);
                const uint32_t h = Fingerprint(w, 2)[0] & mask;
                home = home == UINT32_MAX ? h : home;
                if (h == home)
                {
                    keys.push_back(w);
                }
            }
            std::vector<Words> inputs;
            std::vector<size_t> key_of;
            for (int repeat = 0; repeat < 64; ++repeat)
            {
                for (size_t k = 0; k < keys.size(); ++k)
                {
                    inputs.push_back(keys[k]);
                    key_of.push_back(k);
                }
            }
            const std::vector<Result> results = t.Run(kModeInsert, 2, inputs);
            const auto payload_of = CheckInserts(t, keys, key_of, results);
            REQUIRE(payload_of.size() == caelix::kHashProbes);
            const HashCounters c = t.Counters();
            REQUIRE(c.inserted == caelix::kHashProbes);
            REQUIRE(c.failed_slot > 0);
            const HashCounters m = t.Maintain(3, 128);
            CheckRebuilt(t, 3, m);
            REQUIRE(m.live == caelix::kHashProbes);
            CheckPartition(t);
        }
        // 2. Two keys with the same key_lo (and so the same home slot) but different key_hi, found by a
        // birthday search: present with two payloads after one frame.
        {
            std::unordered_map<uint32_t, Words> by_lo;
            Words a{};
            Words b{};
            bool found = false;
            while (!found)
            {
                const Words w = RandomWords(rng, 2);
                const auto fp = Fingerprint(w, 2);
                const auto [it, added] = by_lo.emplace(fp[0], w);
                if (!added && Fingerprint(it->second, 2)[1] != fp[1])
                {
                    a = it->second;
                    b = w;
                    found = true;
                }
            }
            Table t(rig.gpu, 1024, 2, 4096);
            t.Maintain(2, 128);
            const std::vector<Words> keys = {a, b};
            std::vector<Words> inputs;
            std::vector<size_t> key_of;
            for (int repeat = 0; repeat < 1024; ++repeat)
            {
                inputs.push_back(keys[repeat & 1]);
                key_of.push_back(repeat & 1);
            }
            // Frame 2: at least one is inserted; the other may be pending (it met the first's insert).
            const auto first = CheckInserts(t, keys, key_of, t.Run(kModeInsert, 2, inputs));
            REQUIRE(!first.empty());
            // A second dispatch of the same frame finds the first published and inserts the other in
            // the next slot.
            const auto second = CheckInserts(t, keys, key_of, t.Run(kModeInsert, 2, inputs));
            REQUIRE(second.size() == 2);
            REQUIRE(second.at(0) != second.at(1));
            const HashCounters m = t.Maintain(3, 128);
            CheckRebuilt(t, 3, m);
            REQUIRE(m.live == 2);
            const std::vector<Result> look = t.Run(kModeLookup, 3, keys);
            REQUIRE(look[0].result == caelix::kHashHit);
            REQUIRE(look[1].result == caelix::kHashHit);
            REQUIRE(look[0].payload == second.at(0));
            REQUIRE(look[1].payload == second.at(1));
            CheckPartition(t);
        }
    }
    test::CloseRig(rig);
}

TEST_CASE("hash table: payload exhaustion, eviction and recovery", "[hash_table]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_hash_table_test", test::RigDesc(0, 0, 0));
    {
        std::mt19937 rng(41);
        constexpr uint32_t kCapacity = 256;
        constexpr uint32_t kMaxAge = 16;
        Table t(rig.gpu, kCapacity, 4, 8192);
        t.Maintain(2, kMaxAge);
        // 600 keys, 8 threads each: the free stack runs empty after 256 inserts.
        std::vector<Words> keys(600);
        for (Words& k : keys)
        {
            k = RandomWords(rng, 4);
        }
        std::vector<Words> inputs;
        std::vector<size_t> key_of;
        for (int repeat = 0; repeat < 8; ++repeat)
        {
            for (size_t k = 0; k < keys.size(); ++k)
            {
                inputs.push_back(keys[k]);
                key_of.push_back(k);
            }
        }
        std::shuffle(inputs.begin(), inputs.end(), rng);  // key_of is rebuilt below
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            for (size_t k = 0; k < keys.size(); ++k)
            {
                if (keys[k] == inputs[i])
                {
                    key_of[i] = k;
                    break;
                }
            }
        }
        const std::vector<Result> results = t.Run(kModeInsert, 2, inputs);
        const auto payload_of = CheckInserts(t, keys, key_of, results);
        const HashCounters c = t.Counters();
        std::printf("600 keys into 256 payloads: %u inserted, %u without payload, %u full windows, top %d\n",
                    c.inserted, c.failed_payload, c.failed_slot, t.Top());
        REQUIRE(c.inserted == kCapacity);
        REQUIRE(payload_of.size() == kCapacity);
        REQUIRE(c.failed_payload > 0);
        REQUIRE(t.Top() < 0);  // the pops went below 0
        // A key published without a payload is an empty entry for the rest of the frame.
        std::set<size_t> without;
        for (size_t i = 0; i < results.size(); ++i)
        {
            if (results[i].result == caelix::kHashNoPayload)
            {
                without.insert(key_of[i]);
            }
        }
        REQUIRE(without.size() == c.failed_payload);
        for (const size_t k : without)
        {
            REQUIRE(t.Run(kModeLookup, 2, {keys[k]})[0].result == caelix::kHashEmptyEntry);
        }
        // The next maintenance: the fix-up clamps top to 0, the empty entries are gone.
        HashCounters m = t.Maintain(3, kMaxAge);
        CheckRebuilt(t, 3, m);
        REQUIRE(t.Top() == 0);
        REQUIRE(m.live == kCapacity);
        for (const size_t k : without)
        {
            REQUIRE(t.Run(kModeLookup, 3, {keys[k]})[0].result == caelix::kHashMissing);
        }
        CheckPartition(t);
        // Inserts while the pool is full: no payload; the stack stays consistent.
        const std::vector<Result> full = t.Run(kModeInsert, 3, {keys[*without.begin()]});
        REQUIRE(full[0].result == caelix::kHashNoPayload);
        // Eviction at the age boundary: touched at 2, kept while frame - 2 <= kMaxAge.
        m = t.Maintain(2 + kMaxAge, kMaxAge);
        REQUIRE(m.evicted == 0);
        REQUIRE(m.live == kCapacity);
        m = t.Maintain(3 + kMaxAge, kMaxAge);
        CheckRebuilt(t, 3 + kMaxAge, m);
        REQUIRE(m.evicted == kCapacity);
        REQUIRE(m.live == 0);
        REQUIRE(t.Top() == static_cast<int32_t>(kCapacity));
        CheckPartition(t);
        // Recovery: the keys that found no payload are inserted now.
        std::vector<Words> later;
        std::vector<size_t> later_of;
        for (const size_t k : without)
        {
            later.push_back(keys[k]);
            later_of.push_back(later_of.size());
        }
        const auto recovered = CheckInserts(t, later, later_of, t.Run(kModeInsert, 3 + kMaxAge, later));
        REQUIRE(recovered.size() == std::min<size_t>(later.size(), kCapacity));
        m = t.Maintain(4 + kMaxAge, kMaxAge);
        CheckRebuilt(t, 4 + kMaxAge, m);
        CheckPartition(t);
    }
    test::CloseRig(rig);
}

TEST_CASE("hash table: random frames keep the pool partitioned", "[hash_table]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_hash_table_test", test::RigDesc(0, 0, 0));
    {
        std::mt19937 rng(51);
        constexpr uint32_t kCapacity = 2048;
        constexpr uint32_t kMaxAge = 6;
        Table t(rig.gpu, kCapacity, 5, 50000);
        std::vector<Words> universe(6000);
        for (Words& k : universe)
        {
            k = RandomWords(rng, 5);
        }
        uint32_t frame = 2;
        HashCounters m = t.Maintain(frame, kMaxAge);
        for (int round = 0; round < 40; ++round)
        {
            // A random subset of the universe, many threads per key (some frames overfill the pool).
            const size_t distinct = 100 + rng() % 3000;
            std::vector<Words> keys;
            std::vector<size_t> key_of;
            std::vector<Words> inputs;
            std::set<size_t> chosen;
            while (chosen.size() < distinct)
            {
                chosen.insert(rng() % universe.size());
            }
            for (const size_t u : chosen)
            {
                keys.push_back(universe[u]);
            }
            const size_t threads = distinct * (1 + rng() % 12);
            for (size_t i = 0; i < threads; ++i)
            {
                key_of.push_back(rng() % keys.size());
                inputs.push_back(keys[key_of.back()]);
            }
            CheckInserts(t, keys, key_of, t.Run(kModeInsert, frame, inputs));
            ++frame;
            m = t.Maintain(frame, kMaxAge);
            INFO("round " << round << " frame " << frame);
            CheckRebuilt(t, frame, m);
            CheckPartition(t);
        }
        std::printf("40 random frames: last maintenance live %u, active %u, evicted %u\n", m.live, m.active,
                    m.evicted);
    }
    test::CloseRig(rig);
}

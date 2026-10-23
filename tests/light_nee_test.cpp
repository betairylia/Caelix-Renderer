// The light NEE (docs/LIGHT_NEE.md) beyond the image test of caelix_trace_test ("light NEE: unbiased
// against the path tracer"): the probe kernel tests/shaders/caelix_light_nee_probe.hlsl runs the light
// cells' and the light tree's functions on a scene with lamps on a plain and on a rotated entity, and the
// public C ABI runs the lifecycle:
//   - cell regions: every receiver that reads a cell (any hit face, both levels, the jitter range and its
//     edges, a rotated entity) lies in the cell's region and has its normal;
//   - support: for receiver boxes (cell regions, boxes touching lamp planes, zero-size boxes), every light
//     face that can send light to a receiver point of the box (both cosines positive for some point of
//     the face) has a positive box pdf and a positive target (the leaf's box importance);
//   - MIS: a lamp point drawn on the NEE side and found again as a BSDF hit gives the same in-brick
//     density and proxy on both sides, and the two balance heuristic weights sum to 1;
//   - lifecycle: a full cell table (inserts without payload) and its recovery after eviction, an entity
//     removed and its id reused at once (a new life), a group released and re-created, a renderer paused
//     while the scene changes many times, the options and their argument checks; no validation or library
//     error, finite images.
#include "test_kernels.h"
#include "test_scenes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#ifndef CAELIX_TEST_SHADER_DIR
#error "CAELIX_TEST_SHADER_DIR must name the build's shader folder"
#endif

namespace {

using test::Brick;
using test::Key;
using test::World;

constexpr uint16_t kGrey = 0x8000u | (15u << 10) | (15u << 5) | 15u;             // opaque rgb5
constexpr uint16_t kLamp = 0x4000u | (15u << 10) | (15u << 6) | (15u << 2) | 3u;  // rgb4 1, strength 3
constexpr uint32_t kGroupShift = 4;
constexpr int32_t kGroupVoxels = 8 << kGroupShift;

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<Vec3, 3>;  // rows

Vec3 Add(const Vec3& a, const Vec3& b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
Vec3 Sub(const Vec3& a, const Vec3& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Vec3 Scale(const Vec3& a, double s)
{
    return {a[0] * s, a[1] * s, a[2] * s};
}
double Dot(const Vec3& a, const Vec3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vec3 Mul(const Mat3& m, const Vec3& v)
{
    return {Dot(m[0], v), Dot(m[1], v), Dot(m[2], v)};
}
Vec3 FaceDir(uint32_t f)
{
    Vec3 d{0.0, 0.0, 0.0};
    d[f >> 1] = (f & 1) != 0 ? -1.0 : 1.0;
    return d;
}

// The library's rotation of a quaternion (x, y, z, w): rows of R.
Mat3 Rotation(const float q[4])
{
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    return {{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
             {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
             {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}}};
}

int32_t FloorDiv(int32_t v, int32_t d)
{
    return v >= 0 ? v / d : -((-v + d - 1) / d);
}

// The index of the voxel (x, y, z) (entity-local) in the brick `key`.
size_t BrickIndex(const Key& key, int32_t x, int32_t y, int32_t z)
{
    return static_cast<size_t>((x - key[0] * 8) + 8 * (y - key[1] * 8) + 64 * (z - key[2] * 8));
}

void SetVoxel(World& world, int32_t x, int32_t y, int32_t z, uint16_t value)
{
    const Key key{FloorDiv(x, 8), FloorDiv(y, 8), FloorDiv(z, 8)};
    auto found = world.bricks.find(key);
    if (found == world.bricks.end())
    {
        found = world.bricks.emplace(key, Brick{}).first;
    }
    found->second[BrickIndex(key, x, y, z)] = value;
}

uint16_t VoxelAt(const World& world, int32_t x, int32_t y, int32_t z)
{
    const Key key{FloorDiv(x, 8), FloorDiv(y, 8), FloorDiv(z, 8)};
    const auto found = world.bricks.find(key);
    if (found == world.bricks.end())
    {
        return 0;
    }
    return found->second[BrickIndex(key, x, y, z)];
}

// Lamps (single voxels and small clusters) and grey blocks around the origin, across group borders.
World LampWorld(std::mt19937& rng, int32_t extent)
{
    World world;
    std::uniform_int_distribution<int32_t> coord(-extent, extent - 1);
    for (int i = 0; i < 40; ++i)
    {
        const int32_t x = coord(rng), y = coord(rng), z = coord(rng);
        const int32_t size = 1 + static_cast<int32_t>(rng() % 3);
        for (int32_t dz = 0; dz < size; ++dz)
        {
            for (int32_t dy = 0; dy < size; ++dy)
            {
                for (int32_t dx = 0; dx < size; ++dx)
                {
                    SetVoxel(world, x + dx, y + dy, z + dz, (rng() % 4) != 0 ? kLamp : kGrey);
                }
            }
        }
    }
    return world;
}

struct EntityModel
{
    CaelixEntityId id = CAELIX_INVALID_ENTITY;
    World world;
    Mat3 rotation{};
    Vec3 translation{};
};

// A light face: an open face of a lamp voxel (the neighbour in its direction is not opaque).
struct LightFace
{
    uint32_t entity_index;
    int32_t voxel[3];  // entity-local
    uint32_t face;
    uint32_t group;
    uint32_t voxel_in_group;  // packed x | y << 10 | z << 20
    Vec3 centre;              // world
    Vec3 normal;              // world
    std::array<Vec3, 4> corners;
};

bool Opaque(uint16_t v)
{
    return (v & 0xC000u) != 0;  // packed encoding
}

std::vector<LightFace> LightFaces(const test::Rig& rig, const std::vector<EntityModel>& entities)
{
    std::vector<LightFace> faces;
    for (uint32_t e = 0; e < entities.size(); ++e)
    {
        const EntityModel& m = entities[e];
        for (const auto& [key, brick] : m.world.bricks)
        {
            for (int32_t i = 0; i < 512; ++i)
            {
                if (brick[static_cast<size_t>(i)] != kLamp)
                {
                    continue;
                }
                const int32_t v[3] = {key[0] * 8 + (i & 7), key[1] * 8 + ((i >> 3) & 7),
                                      key[2] * 8 + (i >> 6)};
                CaelixBrickInfo info{};
                const CaelixBrickKey brick_key{key[0], key[1], key[2]};
                REQUIRE(caelix_scene_query_brick(rig.scene, m.id, brick_key, &info) == CAELIX_OK);
                for (uint32_t f = 0; f < 6; ++f)
                {
                    const Vec3 d = FaceDir(f);
                    const int32_t nx = v[0] + static_cast<int32_t>(d[0]);
                    const int32_t ny = v[1] + static_cast<int32_t>(d[1]);
                    const int32_t nz = v[2] + static_cast<int32_t>(d[2]);
                    if (Opaque(VoxelAt(m.world, nx, ny, nz)))
                    {
                        continue;
                    }
                    LightFace lf;
                    lf.entity_index = e;
                    std::copy(v, v + 3, lf.voxel);
                    lf.face = f;
                    lf.group = info.group_index;
                    uint32_t packed = 0;
                    for (int a = 0; a < 3; ++a)
                    {
                        const int32_t origin = FloorDiv(v[a], kGroupVoxels) * kGroupVoxels;
                        packed |= static_cast<uint32_t>(v[a] - origin) << (10 * a);
                    }
                    lf.voxel_in_group = packed;
                    const Vec3 local_centre = Add({v[0] + 0.5, v[1] + 0.5, v[2] + 0.5}, Scale(d, 0.5));
                    lf.centre = Add(Mul(m.rotation, local_centre), m.translation);
                    lf.normal = Mul(m.rotation, d);
                    const uint32_t axis = f >> 1;
                    const uint32_t b = (axis + 1) % 3;
                    const uint32_t c = (axis + 2) % 3;
                    int k = 0;
                    for (const double sb : {-0.5, 0.5})
                    {
                        for (const double sc : {-0.5, 0.5})
                        {
                            Vec3 corner = local_centre;
                            corner[b] += sb;
                            corner[c] += sc;
                            lf.corners[static_cast<size_t>(k++)] =
                                Add(Mul(m.rotation, corner), m.translation);
                        }
                    }
                    faces.push_back(lf);
                }
            }
        }
    }
    return faces;
}

// A face can send light to the receiver x with normal n: it faces x (n_f . (x - c) > 0, constant over
// the face) and a point of the face lies in front of the receiver (n . (y - x) > 0 at a corner).
bool Contributes(const LightFace& face, const Vec3& x, const Vec3& n)
{
    if (!(Dot(face.normal, Sub(x, face.centre)) > 1e-9))
    {
        return false;
    }
    for (const Vec3& corner : face.corners)
    {
        if (Dot(n, Sub(corner, x)) > 1e-9)
        {
            return true;
        }
    }
    return false;
}

// Push constants of tests/shaders/caelix_light_nee_probe.hlsl (the same member names).
struct ProbePush
{
    uint64_t descriptor;
    uint64_t input;
    uint64_t input2;
    uint64_t output;
    uint32_t mode;
    uint32_t count;
    uint32_t draws;
    uint32_t seed;
};
static_assert(sizeof(ProbePush) == 48);

struct alignas(16) RegionQuery
{
    uint32_t entity, face, level, jitter;
    float u[3];
    float pad0;
    float hit[3];
    float pad1;
    float pad2[4];
};
static_assert(sizeof(RegionQuery) == 64);

struct alignas(16) SupportQuery
{
    float lo[3];
    float front_x;  // a point in front of the light face
    float hi[3];
    float front_y;
    float normal[3];
    float front_z;
    uint32_t group, voxel, face, block;
};
static_assert(sizeof(SupportQuery) == 64);

struct alignas(16) MisQuery
{
    float lo[3];
    float norm;
    float hi[3];
    float pad0;
    float normal[3];
    float pad1;
    float x[3];
    float pad2;
};
static_assert(sizeof(MisQuery) == 64);

struct Box
{
    Vec3 lo;
    Vec3 hi;
    Vec3 normal;
};

class Probe
{
public:
    explicit Probe(test::Gpu& gpu)
        : gpu_(gpu), kernel_(gpu, CAELIX_TEST_SHADER_DIR, "caelix_light_nee_probe", sizeof(ProbePush))
    {
    }

    template <typename Query, typename Result>
    std::vector<Result> Run(uint64_t descriptor, uint32_t mode, const std::vector<Query>& queries,
                            uint32_t draws = 1, uint32_t seed = 0)
    {
        const uint32_t count = static_cast<uint32_t>(queries.size()) * draws;
        test::HostBuffer in = test::CreateHostBuffer(gpu_, queries.size() * sizeof(Query));
        test::HostBuffer out = test::CreateHostBuffer(gpu_, uint64_t{count} * sizeof(Result));
        std::memcpy(in.data, queries.data(), queries.size() * sizeof(Query));
        ProbePush push{descriptor, in.address, in.address, out.address, mode, count, draws, seed};
        test::SubmitAndWait(gpu_,
                            [&](VkCommandBuffer cmd) { kernel_.Dispatch(cmd, &push, (count + 63) / 64); });
        std::vector<Result> results(count);
        std::memcpy(results.data(), out.data, results.size() * sizeof(Result));
        test::DestroyHostBuffer(gpu_, in);
        test::DestroyHostBuffer(gpu_, out);
        return results;
    }

private:
    test::Gpu& gpu_;
    test::ComputeKernel kernel_;
};

// The scene of the probe cases: entity 0 (the rig's, identity) and a rotated, translated entity 1, both
// with lamps; the light tree built.
struct LampScene
{
    test::Rig rig;
    std::vector<EntityModel> entities;
    std::vector<LightFace> faces;
    uint64_t descriptor = 0;
};

void OpenLampScene(LampScene& s, uint32_t seed)
{
    test::OpenRig(s.rig, "caelix_light_nee_test", test::RigDesc(0, kGroupShift, 0));
    const CaelixLightOptions light{1, 0.0f, 0};
    REQUIRE(caelix_scene_set_light_options(s.rig.scene, &light) == CAELIX_OK);
    std::mt19937 rng(seed);
    EntityModel a;
    a.id = s.rig.entity;
    a.world = LampWorld(rng, 40);
    const float identity[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    a.rotation = Rotation(identity);
    a.translation = {0.0, 0.0, 0.0};
    EntityModel b;
    CaelixEntityDesc desc{};
    desc.position[0] = 150.0f;
    desc.position[1] = 20.0f;
    desc.position[2] = -30.0f;
    // 37 degrees about (1, 2, 3).
    const double angle = 37.0 * 3.14159265358979323846 / 180.0;
    const double axis_len = std::sqrt(14.0);
    for (int i = 0; i < 3; ++i)
    {
        desc.rotation[i] = static_cast<float>(std::sin(angle / 2) * (i + 1) / axis_len);
    }
    desc.rotation[3] = static_cast<float>(std::cos(angle / 2));
    REQUIRE(caelix_scene_add_entity(s.rig.scene, &desc, &b.id) == CAELIX_OK);
    b.world = LampWorld(rng, 24);
    b.rotation = Rotation(desc.rotation);
    b.translation = {desc.position[0], desc.position[1], desc.position[2]};
    s.entities = {a, b};
    for (const EntityModel& m : s.entities)
    {
        const CaelixEntityId saved = s.rig.entity;
        s.rig.entity = m.id;
        test::Upload(s.rig, m.world, test::KeysOf(m.world));
        s.rig.entity = saved;
    }
    for (int i = 0; i < 4; ++i)
    {
        test::FlushFrame(s.rig.gpu, s.rig.scene, s.rig.frames);
    }
    s.descriptor = caelix_scene_get_light_tree(s.rig.scene);
    REQUIRE(s.descriptor != 0);
    s.faces = LightFaces(s.rig, s.entities);
    REQUIRE(s.faces.size() > 100);
}

// A random hit on a voxel face of entity `e` near its lamps: (local hit point, face).
struct Hit
{
    uint32_t entity_index;
    Vec3 local;
    uint32_t face;
};

Hit RandomHit(std::mt19937& rng, uint32_t entity_index, int32_t extent)
{
    std::uniform_int_distribution<int32_t> coord(-extent, extent);
    std::uniform_real_distribution<double> t(-0.5, 0.5);
    Hit h;
    h.entity_index = entity_index;
    h.face = rng() % 6;
    const Vec3 d = FaceDir(h.face);
    Vec3 p = {coord(rng) + 0.5, coord(rng) + 0.5, coord(rng) + 0.5};
    p = Add(p, Scale(d, 0.5));
    for (int a = 0; a < 3; ++a)
    {
        if (d[a] == 0.0)
        {
            p[a] += t(rng);
        }
    }
    h.local = p;
    return h;
}

}  // namespace

TEST_CASE("light NEE: cell regions hold every receiver that reads them", "[light_nee]")
{
    LampScene s;
    OpenLampScene(s, 7);
    {
        Probe probe(s.rig.gpu);
        std::mt19937 rng(11);
        std::vector<RegionQuery> queries;
        std::vector<Hit> hits;
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        for (int i = 0; i < 20000; ++i)
        {
            const uint32_t e = static_cast<uint32_t>(i & 1);
            const Hit h = RandomHit(rng, e, 300);
            RegionQuery q{};
            q.entity = s.entities[e].id;
            q.face = h.face;
            q.level = static_cast<uint32_t>(rng() % 2);
            q.jitter = (i % 5) != 0 ? 1u : 0u;
            for (int a = 0; a < 3; ++a)
            {
                // The edges of the jitter range at the start: 0 and the largest float below 1.
                const float edge = (i % 3 == 0) ? 0.0f : 0.99999994f;
                q.u[a] = i < 600 ? edge : u01(rng);
                q.hit[a] = static_cast<float>(h.local[a]);
            }
            queries.push_back(q);
            hits.push_back(h);
        }
        struct RegionResult
        {
            float lo[3];
            float hi[3];
            float normal[3];
            int32_t coord[3];
        };
        static_assert(sizeof(RegionResult) == 48);
        const std::vector<RegionResult> results =
            probe.Run<RegionQuery, RegionResult>(s.descriptor, 0, queries);
        double worst_inside = 1e30;
        for (size_t i = 0; i < queries.size(); ++i)
        {
            const RegionQuery& q = queries[i];
            const RegionResult& r = results[i];
            const EntityModel& m = s.entities[hits[i].entity_index];
            // The CPU cell rule (LightCellCoordOfHit).
            const double edge = static_cast<double>(1 << (5 + 2 * q.level));
            const Vec3 d = FaceDir(q.face);
            for (int a = 0; a < 3; ++a)
            {
                double p = static_cast<double>(q.hit[a]) + 0.25 * d[a];
                if (q.jitter != 0)
                {
                    p += (static_cast<double>(q.u[a]) - 0.5) * edge;
                }
                const int32_t expected = static_cast<int32_t>(std::floor(p)) >> (5 + 2 * q.level);
                INFO("query " << i << " axis " << a);
                CHECK(r.coord[a] == expected);
            }
            // The receiver: the world hit point pushed 0.002 off the face (K_RAY_ORIGIN_PUSH_OFF).
            const Vec3 hit_world = Add(Mul(m.rotation, {q.hit[0], q.hit[1], q.hit[2]}), m.translation);
            const Vec3 n = Mul(m.rotation, d);
            const Vec3 x = Add(hit_world, Scale(n, 0.002));
            for (int a = 0; a < 3; ++a)
            {
                const double inside = std::min(x[a] - r.lo[a], r.hi[a] - x[a]);
                worst_inside = std::min(worst_inside, inside);
                INFO("query " << i << " axis " << a << " x " << x[a] << " lo " << r.lo[a] << " hi "
                              << r.hi[a]);
                REQUIRE(inside > 0.0);
                REQUIRE(std::abs(r.normal[a] - n[a]) < 1e-5);
            }
        }
        std::printf("cell regions: %zu receivers inside their region, smallest margin %.4f voxels\n",
                    queries.size(), worst_inside);
    }
    test::CloseRig(s.rig);
}

TEST_CASE("light NEE: the box sampler supports every light face that can reach the box", "[light_nee]")
{
    LampScene s;
    OpenLampScene(s, 9);
    {
        Probe probe(s.rig.gpu);
        std::mt19937 rng(13);
        // Receiver boxes: cell regions of random hits near the lamps (both levels, both entities), the
        // same boxes shrunk to a point (the point form), and thin boxes on the planes of lamp faces.
        std::vector<Box> boxes;
        std::vector<RegionQuery> regions;
        for (int i = 0; i < 160; ++i)
        {
            const uint32_t e = static_cast<uint32_t>(i & 1);
            const Hit h = RandomHit(rng, e, 48);
            RegionQuery q{};
            q.entity = s.entities[e].id;
            q.face = h.face;
            q.level = static_cast<uint32_t>(i % 4 == 0);
            q.jitter = 1;
            q.u[0] = q.u[1] = q.u[2] = 0.5f;
            for (int a = 0; a < 3; ++a)
            {
                q.hit[a] = static_cast<float>(h.local[a]);
            }
            regions.push_back(q);
        }
        struct RegionResult
        {
            float lo[3];
            float hi[3];
            float normal[3];
            int32_t coord[3];
        };
        for (const RegionResult& r : probe.Run<RegionQuery, RegionResult>(s.descriptor, 0, regions))
        {
            boxes.push_back({{r.lo[0], r.lo[1], r.lo[2]}, {r.hi[0], r.hi[1], r.hi[2]},
                             {r.normal[0], r.normal[1], r.normal[2]}});
        }
        const size_t region_boxes = boxes.size();
        for (size_t i = 0; i < region_boxes; i += 2)
        {
            const Vec3 c = Scale(Add(boxes[i].lo, boxes[i].hi), 0.5);
            boxes.push_back({c, c, boxes[i].normal});  // zero size
        }
        for (size_t i = 0; i < s.faces.size(); i += 17)
        {
            // A receiver box whose face lies in the lamp face's plane, facing along the lamp's normal.
            const LightFace& f = s.faces[i];
            Box b;
            b.lo = Sub(f.centre, {3.0, 3.0, 3.0});
            b.hi = Add(f.centre, {3.0, 3.0, 3.0});
            b.normal = f.normal;
            boxes.push_back(b);
        }
        std::vector<SupportQuery> queries;
        std::vector<std::pair<size_t, size_t>> pairs;  // (box, face) of every query
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        size_t considered = 0;
        for (size_t bi = 0; bi < boxes.size(); ++bi)
        {
            const Box& b = boxes[bi];
            // Receiver points: the 8 corners, the centre and 24 random points of the box.
            std::vector<Vec3> points;
            for (int k = 0; k < 8; ++k)
            {
                points.push_back({(k & 1) ? b.hi[0] : b.lo[0], (k & 2) ? b.hi[1] : b.lo[1],
                                  (k & 4) ? b.hi[2] : b.lo[2]});
            }
            points.push_back(Scale(Add(b.lo, b.hi), 0.5));
            for (int k = 0; k < 24; ++k)
            {
                Vec3 point;
                for (int a = 0; a < 3; ++a)
                {
                    point[a] = b.lo[a] + u01(rng) * (b.hi[a] - b.lo[a]);
                }
                points.push_back(point);
            }
            for (size_t fi = 0; fi < s.faces.size(); ++fi)
            {
                ++considered;
                const LightFace& f = s.faces[fi];
                const bool reaches = std::any_of(points.begin(), points.end(),
                                                 [&](const Vec3& x) { return Contributes(f, x, b.normal); });
                if (!reaches)
                {
                    continue;
                }
                SupportQuery q{};
                for (int a = 0; a < 3; ++a)
                {
                    q.lo[a] = static_cast<float>(b.lo[a]);
                    q.hi[a] = static_cast<float>(b.hi[a]);
                    q.normal[a] = static_cast<float>(b.normal[a]);
                }
                const Vec3 front = Add(f.centre, Scale(f.normal, 0.5));
                q.front_x = static_cast<float>(front[0]);
                q.front_y = static_cast<float>(front[1]);
                q.front_z = static_cast<float>(front[2]);
                q.group = f.group;
                q.voxel = f.voxel_in_group;
                q.face = f.face;
                q.block = kLamp;
                queries.push_back(q);
                pairs.emplace_back(bi, fi);
            }
        }
        struct SupportResult
        {
            float found, has_leaf, pdf, target;
        };
        const std::vector<SupportResult> results =
            probe.Run<SupportQuery, SupportResult>(s.descriptor, 1, queries);
        for (size_t i = 0; i < results.size(); ++i)
        {
            const SupportResult& r = results[i];
            const LightFace& f = s.faces[pairs[i].second];
            INFO("box " << pairs[i].first << " face " << pairs[i].second << " (entity " << f.entity_index
                        << ", voxel " << f.voxel[0] << " " << f.voxel[1] << " " << f.voxel[2] << ", face "
                        << f.face << ")");
            REQUIRE(r.found == 1.0f);
            REQUIRE(r.has_leaf == 1.0f);
            REQUIRE(r.pdf > 0.0f);
            REQUIRE(r.target > 0.0f);
        }
        std::printf("support: %zu boxes, %zu light faces, %zu (box, face) pairs that reach the box, all "
                    "with a positive pdf and target (of %zu pairs)\n",
                    boxes.size(), s.faces.size(), results.size(), considered);
    }
    test::CloseRig(s.rig);
}

TEST_CASE("light NEE: both MIS sides see the same proxy and their weights sum to 1", "[light_nee]")
{
    LampScene s;
    OpenLampScene(s, 21);
    {
        Probe probe(s.rig.gpu);
        std::mt19937 rng(23);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        std::vector<MisQuery> queries;
        for (int i = 0; i < 400; ++i)
        {
            const uint32_t e = static_cast<uint32_t>(i & 1);
            const Hit h = RandomHit(rng, e, 48);
            RegionQuery q{};
            q.entity = s.entities[e].id;
            q.face = h.face;
            q.level = static_cast<uint32_t>(i % 3 == 0);
            q.jitter = 1;
            q.u[0] = static_cast<float>(u01(rng));
            q.u[1] = static_cast<float>(u01(rng));
            q.u[2] = static_cast<float>(u01(rng));
            for (int a = 0; a < 3; ++a)
            {
                q.hit[a] = static_cast<float>(h.local[a]);
            }
            struct RegionResult
            {
                float lo[3];
                float hi[3];
                float normal[3];
                int32_t coord[3];
            };
            const RegionResult r =
                probe.Run<RegionQuery, RegionResult>(s.descriptor, 0, std::vector<RegionQuery>{q})[0];
            const EntityModel& m = s.entities[e];
            const Vec3 hit_world = Add(Mul(m.rotation, h.local), m.translation);
            const Vec3 x = Add(hit_world, Scale(Mul(m.rotation, FaceDir(h.face)), 0.002));
            MisQuery mq{};
            for (int a = 0; a < 3; ++a)
            {
                mq.lo[a] = r.lo[a];
                mq.hi[a] = r.hi[a];
                mq.normal[a] = r.normal[a];
                mq.x[a] = static_cast<float>(x[a]);
            }
            mq.norm = static_cast<float>(0.01 + u01(rng) * 10.0);
            queries.push_back(mq);
        }
        struct MisResult
        {
            float ok, same, pdf, in_brick, p_nee, p_bsdf, w_nee, w_bsdf;
        };
        constexpr uint32_t kDraws = 64;
        const std::vector<MisResult> results =
            probe.Run<MisQuery, MisResult>(s.descriptor, 2, queries, kDraws, 99);
        size_t ok = 0;
        size_t with_proxy = 0;
        double worst_sum = 0.0;
        for (size_t i = 0; i < results.size(); ++i)
        {
            const MisResult& r = results[i];
            if (r.ok == 0.0f)
            {
                continue;
            }
            ++ok;
            with_proxy += r.p_nee > 0.0f ? 1u : 0u;
            INFO("draw " << i << " pdf " << r.pdf << " in_brick " << r.in_brick << " p_nee " << r.p_nee
                         << " p_bsdf " << r.p_bsdf << " w " << r.w_nee << " + " << r.w_bsdf);
            REQUIRE(r.same == 1.0f);
            REQUIRE(r.in_brick == r.pdf);
            REQUIRE(r.p_bsdf == r.p_nee);
            const double sum = static_cast<double>(r.w_nee) + r.w_bsdf;
            worst_sum = std::max(worst_sum, std::abs(sum - 1.0));
            REQUIRE(std::abs(sum - 1.0) < 1e-6);
        }
        std::printf("MIS: %zu drawn lamp points of %zu draws (%zu with a proxy), largest "
                    "|w_nee + w_bsdf - 1| = %.2e\n",
                    ok, results.size(), with_proxy, worst_sum);
        REQUIRE(ok > results.size() / 2);
        REQUIRE(with_proxy > ok / 2);
    }
    test::CloseRig(s.rig);
}

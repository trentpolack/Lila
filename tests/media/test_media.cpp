#include <doctest/doctest.h>

#include "TestClips.h"

#include "media/AudioEngine.h"
#include "media/Media.h"
#include "ops/Command.h"
#include "scene/Anim.h"
#include "scene/Graph.h"
#include "scene/ImagePlane.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using json = nlohmann::json;

namespace
{
using namespace lila;
using scene::Graph;

json mediaRun(const char* cmd, const json& call = json::object())
{
    return ops::execute(cmd, call);
}

scene::NodeId mediaId(const std::string& name)
{
    scene::Node* n = Graph::current().find(name);
    REQUIRE(n);
    return n->id();
}

/** Two whole-scene takes: Walk (0 to 2 s) and Run (3 to 5 s), Walk active. */
void mediaTwoTakes()
{
    mediaRun("newScene");
    mediaRun("playbackOptions", {{"fps", 24.0}});
    mediaRun("createTake", {{"name", "Walk"}, {"start", 0.0}, {"end", 2.0}, {"fps", 24.0}});
    mediaRun("createTake", {{"name", "Run"}, {"start", 3.0}, {"end", 5.0}, {"fps", 24.0}});
}
} // namespace

TEST_CASE("take media: a take's audio and video play only while it is active, timed from its start")
{
    mediaTwoTakes();
    const std::string wav = mediatest::writeWav("walk.wav", 48000, 1.0, 0.25).string();
    const std::string clip = mediatest::writeClip("run.mp4", 10, 24, 5).string();
    CHECK(mediaRun("audio", {{"fileName", wav}, {"offset", 12.0}, {"take", "Walk"}, {"name", "walkVO"}}) == "walkVO");
    mediaRun("audio", {{"fileName", wav}, {"name", "music"}});
    mediaRun("imagePlane", {{"fileName", clip}, {"name", "runRef"}, {"startFrame", 6.0}});
    mediaRun("bindMedia", {{"args", {"Run", "runRef"}}});

    Graph& g = Graph::current();
    CHECK(media::isLive(g, mediaId("walkVO")));
    CHECK_FALSE(media::isLive(g, mediaId("runRef")));
    CHECK(media::isLive(g, mediaId("music")));
    // Walk starts at 0 s: offset 12 frames at 24 fps is half a second in.
    CHECK(media::startSeconds(g, mediaId("walkVO"), 12.0) == doctest::Approx(0.5));
    CHECK(media::audibleSounds(g).size() == 2);

    mediaRun("setActiveTake", {{"args", {"Run"}}});
    CHECK_FALSE(media::isLive(g, mediaId("walkVO")));
    CHECK(media::isLive(g, mediaId("runRef")));
    CHECK(media::audibleSounds(g) == std::vector<scene::NodeId>{mediaId("music")});
    // Run starts at 3 s; the plane's first picture is 6 frames later.
    CHECK(media::startSeconds(g, mediaId("runRef"), 6.0) == doctest::Approx(3.25));
    media::VideoTiming t{media::startSeconds(g, mediaId("runRef"), 6.0), 0.0, 0};
    CHECK(media::videoFrame(t, 24.0, 3.25 + 4.0 / 24.0) == 4);

    json takes = mediaRun("listTakes");
    REQUIRE(takes.size() == 2);
    CHECK(takes[0]["media"] == json::array({"walkVO"}));
    CHECK(takes[1]["media"] == json::array({"runRef"}));

    // A copy of a take keeps its media; unbinding frees it to play everywhere.
    mediaRun("duplicateTake", {{"args", {"Walk"}}, {"name", "Walk2"}});
    CHECK(mediaRun("listTakes")[2]["media"] == json::array({"walkVO"}));
    mediaRun("bindMedia", {{"args", {"Walk", "walkVO"}}, {"remove", true}});
    mediaRun("bindMedia", {{"args", {"Walk2", "walkVO"}}, {"remove", true}});
    CHECK(media::isLive(g, mediaId("walkVO")));
    mediaRun("undo");
    mediaRun("undo");
    CHECK_FALSE(media::isLive(g, mediaId("walkVO")));

    // Bindings survive a save and reload.
    const std::string path = (mediatest::scratch() / "takemedia.lila").string();
    mediaRun("saveScene", {{"args", {path}}});
    mediaRun("newScene");
    mediaRun("openScene", {{"args", {path}}});
    CHECK(mediaRun("listTakes")[1]["media"] == json::array({"runRef"}));
    CHECK(media::isLive(Graph::current(), mediaId("runRef")));
    CHECK_FALSE(media::isLive(Graph::current(), mediaId("walkVO")));
}

TEST_CASE("time slider sound: one picked sound plays alone, empty plays them all")
{
    mediaTwoTakes();
    const std::string wav = mediatest::writeWav("pick.wav", 48000, 0.5, 0.1).string();
    mediaRun("audio", {{"fileName", wav}, {"name", "a"}});
    mediaRun("audio", {{"fileName", wav}, {"name", "b"}});
    Graph& g = Graph::current();
    CHECK(media::audibleSounds(g).size() == 2);
    mediaRun("timeSliderSound", {{"sound", "b"}});
    CHECK(mediaRun("timeSliderSound") == "b");
    CHECK(media::audibleSounds(g) == std::vector<scene::NodeId>{mediaId("b")});
    mediaRun("timeSliderSound", {{"sound", ""}});
    mediaRun("audio", {{"args", {"a"}}, {"mute", true}});
    CHECK(media::audibleSounds(g) == std::vector<scene::NodeId>{mediaId("b")});
    CHECK_THROWS_AS(mediaRun("timeSliderSound", {{"sound", "Walk"}}), CommandError);
}

TEST_CASE("mix: voices land at their start time, scaled by gain, playing at speed")
{
    auto buf = std::make_shared<media::AudioBuffer>();
    buf->rate = 1000;
    buf->channels = 2;
    for (int i = 0; i < 1000; ++i)
    {
        buf->samples.push_back(float(i) / 1000.0f);
        buf->samples.push_back(-float(i) / 1000.0f);
    }
    std::vector<media::Voice> v = {{buf, 2.0, 0.5, "x"}};
    std::vector<float> out(8);
    media::mix(v, 2.5, 1.0, 1000, out.data(), 4);
    CHECK(out[0] == doctest::Approx(0.25f));
    CHECK(out[1] == doctest::Approx(-0.25f));
    CHECK(out[6] == doctest::Approx(0.5f * 0.503f));
    // Before the start and past the end is silence.
    media::mix(v, 1.0, 1.0, 1000, out.data(), 4);
    CHECK(out[0] == 0.0f);
    // Backwards play reads the samples in reverse.
    media::mix(v, 2.5, -1.0, 1000, out.data(), 4);
    CHECK(out[2] == doctest::Approx(0.5f * 0.499f));
}

TEST_CASE("audio engine: plays from the time given, scrub stops by itself, one-shots mix in")
{
    media::AudioEngine& e = media::AudioEngine::get();
    auto buf = std::make_shared<media::AudioBuffer>();
    buf->rate = media::kSampleRate;
    buf->samples.assign(size_t(media::kSampleRate) * 2, 0.25f);
    e.setVoices({{buf, 0.0, 1.0, "x"}});
    std::vector<float> out(256 * 2);

    e.play(0.5, 1.0);
    e.render(out.data(), 256);
    CHECK(out[0] == doctest::Approx(0.25f));
    CHECK(e.playing());
    // No device is open in tests, so there is no clock to follow.
    CHECK_FALSE(e.clock());

    e.stop();
    e.render(out.data(), 256);
    CHECK(out[0] == 0.0f);

    e.scrub(0.5, 100.0 / media::kSampleRate);
    e.render(out.data(), 256);
    CHECK(out[99 * 2] == doctest::Approx(0.25f));
    CHECK(out[100 * 2] == 0.0f);
    CHECK_FALSE(e.playing());

    auto shot = std::make_shared<media::AudioBuffer>();
    shot->samples.assign(20, 0.5f);
    e.fire(shot, 0.5);
    e.render(out.data(), 256);
    CHECK(out[0] == doctest::Approx(0.25f));
    CHECK(out[20] == 0.0f);
    e.setVoices({});
}

TEST_CASE("audio engine: publishes complete snapshots while rendering")
{
    media::AudioEngine& e = media::AudioEngine::get();
    auto quiet = std::make_shared<media::AudioBuffer>();
    quiet->rate = media::kSampleRate;
    quiet->samples.assign(size_t(media::kSampleRate) * 2, 0.25f);
    auto loud = std::make_shared<media::AudioBuffer>(*quiet);
    loud->samples.assign(quiet->samples.size(), 0.5f);
    e.setVoices({{quiet, 0.0, 1.0, "x"}});
    e.play(0.0);

    std::atomic<bool> start{false};
    std::thread publisher([&] {
        while (!start.load())
            std::this_thread::yield();
        for (int i = 0; i < 2000; ++i)
        {
            e.setVoices({{i % 2 ? quiet : loud, 0.0, 1.0, "x"}});
            if (i % 3)
                e.play(0.0);
            else
                e.stop();
        }
    });
    start.store(true);
    std::vector<float> out(32);
    bool completeSnapshots = true;
    for (int i = 0; i < 2000; ++i)
    {
        e.render(out.data(), 16);
        completeSnapshots &= out[0] == 0.0f || out[0] == 0.25f || out[0] == 0.5f;
        for (float sample : out)
            completeSnapshots &= sample == out[0];
    }
    publisher.join();
    CHECK(completeSnapshots);
    e.stop();
    e.setVoices({});
}

TEST_CASE("audio node: loads its sound and voices follow it")
{
    mediaTwoTakes();
    const std::string wav = mediatest::writeWav("voice.wav", 22050, 0.5, 0.1).string();
    mediaRun("audio", {{"fileName", wav}, {"name", "v"}, {"offset", 24.0}, {"gain", 2.0}});
    std::vector<media::Voice> v = media::voices(Graph::current(), true);
    REQUIRE(v.size() == 1);
    CHECK(v[0].start == doctest::Approx(1.0));
    CHECK(v[0].gain == doctest::Approx(2.0));
    CHECK(v[0].buffer->rate == media::kSampleRate);
    CHECK(v[0].buffer->duration() == doctest::Approx(0.5).epsilon(0.02));
    json list = mediaRun("listMedia");
    REQUIRE(list.size() == 1);
    CHECK(list[0]["type"] == "audio");
    CHECK(list[0]["live"] == true);
}

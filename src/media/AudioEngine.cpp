#include "media/AudioEngine.h"

#include "core/Log.h"

#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include <algorithm>
#include <atomic>
#include <mutex>

namespace lila::media
{
namespace
{
// Apple libc++ still needs the shared_ptr atomic free functions; keep atomic<shared_ptr> where available.
template <class T> using AudioSnapshot =
#if defined(__cpp_lib_atomic_shared_ptr)
    std::atomic<std::shared_ptr<const T>>;
#else
    std::shared_ptr<const T>;
#endif

struct Transport
{
    bool playing = false;
    double time = 0.0;      // scene seconds at anchor
    double speed = 1.0;
    int64_t anchor = 0;     // rendered frame count when it started
    int64_t limit = -1;     // frames to play before going quiet; -1 for no end
};

struct OneShot
{
    AudioBufferPtr buffer;
    float gain = 1.0f;
    int64_t at = 0; // next frame of the buffer
};
} // namespace

struct AudioEngine::Impl
{
    AudioSnapshot<std::vector<Voice>> voices{std::make_shared<const std::vector<Voice>>()};
    AudioSnapshot<Transport> transport{std::make_shared<const Transport>()};
    std::atomic<int64_t> rendered{0};
    std::mutex pendingMutex;
    std::vector<OneShot> pending;
    std::vector<OneShot> active; // the render thread's
    std::vector<float> scratch;

    ma_device device{};
    bool deviceOpen = false;
    bool deviceFailed = false;
    int64_t latency = 0;

    void renderInto(float* out, int frames)
    {
        const std::shared_ptr<const Transport> t = std::atomic_load(&transport);
        const int64_t start = rendered.load();
        std::fill(out, out + size_t(frames) * 2, 0.0f);
        if (t->playing)
        {
            int64_t from = start - t->anchor;
            int count = frames;
            if (t->limit >= 0)
                count = int(std::clamp<int64_t>(t->limit - from, 0, frames));
            if (count > 0)
            {
                const std::shared_ptr<const std::vector<Voice>> v = std::atomic_load(&voices);
                mix(*v, t->time + double(from) / kSampleRate * t->speed, t->speed, kSampleRate, out, count);
            }
        }
        {
            std::unique_lock lk(pendingMutex, std::try_to_lock);
            if (lk.owns_lock() && !pending.empty())
            {
                active.insert(active.end(), pending.begin(), pending.end());
                pending.clear();
            }
        }
        for (OneShot& s : active)
        {
            const AudioBuffer& b = *s.buffer;
            const int64_t n = std::min<int64_t>(frames, b.frames() - s.at);
            for (int64_t i = 0; i < n; ++i)
            {
                out[i * 2] += s.gain * b.samples[size_t((s.at + i) * 2)];
                out[i * 2 + 1] += s.gain * b.samples[size_t((s.at + i) * 2 + 1)];
            }
            s.at += std::max<int64_t>(n, 0);
        }
        std::erase_if(active, [](const OneShot& s) { return s.at >= s.buffer->frames(); });
        for (int i = 0; i < frames * 2; ++i)
            out[i] = std::clamp(out[i], -1.0f, 1.0f);
        rendered.fetch_add(frames);
    }

    void publish(Transport t)
    {
        t.anchor = rendered.load();
        std::atomic_store(&transport, std::make_shared<const Transport>(t));
    }
};

namespace
{
void dataCallback(ma_device* device, void* output, const void*, ma_uint32 frames)
{
    auto* impl = static_cast<AudioEngine::Impl*>(device->pUserData);
    impl->renderInto(static_cast<float*>(output), int(frames));
}
} // namespace

AudioEngine& AudioEngine::get()
{
    static AudioEngine e;
    return e;
}

AudioEngine::AudioEngine() : mImpl(std::make_unique<Impl>()) {}

AudioEngine::~AudioEngine()
{
    close();
}

bool AudioEngine::ensureDevice()
{
    Impl& d = *mImpl;
    if (d.deviceOpen || d.deviceFailed)
        return d.deviceOpen;
    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format = ma_format_f32;
    cfg.playback.channels = 2;
    cfg.sampleRate = kSampleRate;
    cfg.dataCallback = dataCallback;
    cfg.pUserData = &d;
    cfg.performanceProfile = ma_performance_profile_low_latency;
    cfg.periodSizeInMilliseconds = 10;
    if (ma_device_init(nullptr, &cfg, &d.device) != MA_SUCCESS)
    {
        d.deviceFailed = true;
        log::warn("audio: no output device; sound is off");
        return false;
    }
    if (ma_device_start(&d.device) != MA_SUCCESS)
    {
        ma_device_uninit(&d.device);
        d.deviceFailed = true;
        log::warn("audio: the output device won't start; sound is off");
        return false;
    }
    d.latency = int64_t(d.device.playback.internalPeriodSizeInFrames) * std::max<ma_uint32>(1, d.device.playback.internalPeriods);
    d.deviceOpen = true;
    return true;
}

bool AudioEngine::hasDevice() const
{
    return mImpl->deviceOpen;
}

void AudioEngine::close()
{
    if (mImpl && mImpl->deviceOpen)
    {
        ma_device_uninit(&mImpl->device);
        mImpl->deviceOpen = false;
    }
}

void AudioEngine::setVoices(std::vector<Voice> voices)
{
    std::atomic_store(&mImpl->voices, std::make_shared<const std::vector<Voice>>(std::move(voices)));
}

void AudioEngine::play(double sceneTime, double speed)
{
    mImpl->publish({.playing = true, .time = sceneTime, .speed = speed});
}

void AudioEngine::stop()
{
    mImpl->publish({});
}

bool AudioEngine::playing() const
{
    const auto t = std::atomic_load(&mImpl->transport);
    return t->playing && t->limit < 0;
}

void AudioEngine::scrub(double sceneTime, double seconds)
{
    mImpl->publish({.playing = true, .time = sceneTime, .speed = 1.0, .limit = std::max<int64_t>(1, int64_t(seconds * kSampleRate))});
}

void AudioEngine::fire(AudioBufferPtr buffer, double gain)
{
    if (!buffer || buffer->channels != 2 || buffer->frames() == 0)
        return;
    std::lock_guard lk(mImpl->pendingMutex);
    mImpl->pending.push_back({std::move(buffer), float(gain), 0});
}

std::optional<double> AudioEngine::clock() const
{
    const Impl& d = *mImpl;
    if (!d.deviceOpen)
        return std::nullopt;
    const auto t = std::atomic_load(&d.transport);
    if (!t->playing || t->limit >= 0)
        return std::nullopt;
    const int64_t heard = std::max<int64_t>(0, d.rendered.load() - t->anchor - d.latency);
    return t->time + double(heard) / kSampleRate * t->speed;
}

void AudioEngine::render(float* out, int frames)
{
    mImpl->renderInto(out, frames);
}

int64_t AudioEngine::renderedFrames() const
{
    return mImpl->rendered.load();
}
} // namespace lila::media

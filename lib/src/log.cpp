#include "log.h"

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>

namespace caelix {
namespace {

std::mutex g_mutex;
CaelixLogCallback g_callback = nullptr;
void* g_user_data = nullptr;

}  // namespace

void SetLogCallback(CaelixLogCallback callback, void* user_data) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_callback = callback;
  g_user_data = user_data;
}

void log(CaelixLogLevel level, const char* fmt, ...) {
  CaelixLogCallback callback;
  void* user_data;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    callback = g_callback;
    user_data = g_user_data;
  }
  if (callback == nullptr || fmt == nullptr) {
    return;
  }

  // Call the callback outside the lock, so a callback that logs or sets a new
  // callback cannot deadlock.
  char stack_buffer[1024];
  va_list args;
  va_start(args, fmt);
  va_list args_copy;
  va_copy(args_copy, args);
  const int length = std::vsnprintf(stack_buffer, sizeof(stack_buffer), fmt, args);
  va_end(args);

  if (length < 0) {
    va_end(args_copy);
    callback(level, "caelix: log message formatting failed", user_data);
    return;
  }
  if (static_cast<size_t>(length) < sizeof(stack_buffer)) {
    va_end(args_copy);
    callback(level, stack_buffer, user_data);
    return;
  }
  try {
    std::string heap_buffer(static_cast<size_t>(length) + 1, '\0');
    std::vsnprintf(heap_buffer.data(), heap_buffer.size(), fmt, args_copy);
    va_end(args_copy);
    callback(level, heap_buffer.c_str(), user_data);
  } catch (...) {
    va_end(args_copy);
    callback(level, stack_buffer, user_data);  // truncated message
  }
}

}  // namespace caelix

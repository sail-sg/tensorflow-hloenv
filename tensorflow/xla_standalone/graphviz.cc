// Copyright 2022 Garena Online Private Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Non-interactive version of xla/tools/interactive_graphviz
#include <gflags/gflags.h>
#include <stdio.h>
#include <unistd.h>

#include <functional>
#include <sstream>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/tools/hlo_extractor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/command_line_flags.h"

DEFINE_string(hlo, "-", "hlo text file");  // by default read from stdin
DEFINE_string(html, "-", "the path to the generated html file");
DEFINE_string(browser, "/usr/bin/sensible-browser",
              "default browser to view hlo graph.");

namespace xla {
namespace tools {
namespace {

void OpenUrl(absl::string_view url) {
  std::cerr << url << std::endl;

  // If it is a url, try to open it up in the user's browser too.
  if (absl::StartsWithIgnoreCase(url, "http://") ||
      absl::StartsWithIgnoreCase(url, "https://") ||
      absl::StartsWithIgnoreCase(url, "file://")) {
    const char* browser_bin = FLAGS_browser.empty()
                                  ? "/usr/bin/sensible-browser"
                                  : FLAGS_browser.c_str();
    tensorflow::SubProcess p;
    p.SetProgram(browser_bin, {browser_bin, std::string(url)});
    p.Start();
  } else {
    std::cerr << "\nExpected a URL, but got strange graph result (dumped "
                 "above).  If this isn't what you expected, maybe file a bug?"
              << std::endl;
  }
}

void RenderAndDisplayGraph(
    const std::function<StatusOr<std::string>(RenderedGraphFormat)>& renderer) {
  StatusOr<std::string> url_result = renderer(RenderedGraphFormat::kUrl);
  if (url_result.ok()) {
    std::string url = url_result.ValueOrDie();
    OpenUrl(url);
    return;
  }

  // Ignore UNAVAILABLE errors; these are expected when there's no URL renderer
  // plugin registered.
  if (url_result.status().code() != tensorflow::error::UNAVAILABLE) {
    std::cerr << "Unable to render graph as URL: " << url_result.status()
              << std::endl;
    std::cerr << "Trying as HTML..." << std::endl;
  }

  auto* env = tensorflow::Env::Default();
  StatusOr<std::string> html_result = renderer(RenderedGraphFormat::kHtml);
  if (!html_result.ok()) {
    std::cerr << "Failed to render graph as HTML: " << html_result.status()
              << std::endl;
    return;
  }

  if (FLAGS_html == "-") {
    std::cout << html_result.ValueOrDie() << std::endl;
  } else {
    std::string temp_file_path = tensorflow::io::JoinPath(
        ".", absl::StrFormat("%s.%d.html", FLAGS_html, env->NowMicros()));
    auto status = tensorflow::WriteStringToFile(
        env, FLAGS_html, std::move(html_result).ValueOrDie());
    if (status.ok()) {
      OpenUrl(absl::StrCat("file://", temp_file_path));
      return;
    }
    std::cerr << "Failed to write rendered HTML graph to " << temp_file_path
              << ": " << status;
  }
  // We don't bother trying kDot, because kHTML should always work (or if it
  // doesn't, we don't have any reason to believe kDot will work better).
}

void GraphViz(HloModule* module) {
  HloRenderOptions hlo_render_options;
  hlo_render_options.show_backend_config = true;
  hlo_render_options.show_fusion_subcomputations = true;

  const HloComputation* comp = module->entry_computation();
  RenderAndDisplayGraph([&](RenderedGraphFormat format) {
    return xla::RenderGraph(
        *comp, /*label=*/"", comp->parent()->config().debug_options(), format,
        /*hlo_execution_profile=*/nullptr, hlo_render_options);
  });
}

}  // namespace
}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::unique_ptr<xla::HloModule> module;
  if (FLAGS_hlo == "-") {
    std::stringstream ss;
    std::string s;
    while (std::getline(std::cin, s)) {
      ss << s << "\n";
    }
    module = xla::HloRunner::CreateModuleFromString(
                 ss.str(), xla::GetDebugOptionsFromFlags())
                 .ValueOrDie();
  } else {
    module = xla::HloRunner::ReadModuleFromHloTextFile(
                 FLAGS_hlo, xla::GetDebugOptionsFromFlags())
                 .ValueOrDie();
  }
  xla::tools::GraphViz(module.get());
  return 0;
}

import AVKit
import SwiftUI

struct AttachmentMediaPlayer: NSViewRepresentable {
    let url: URL
    let isVideo: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator(url: url)
    }

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .inline
        view.videoGravity = .resizeAspect
        view.showsFullScreenToggleButton = isVideo
        view.player = context.coordinator.player
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if context.coordinator.url != url {
            context.coordinator.url = url
            context.coordinator.player = AVPlayer(url: url)
            view.player = context.coordinator.player
        }
        view.showsFullScreenToggleButton = isVideo
    }

    static func dismantleNSView(_ view: AVPlayerView, coordinator: Coordinator) {
        view.player?.pause()
        view.player = nil
    }

    final class Coordinator {
        var url: URL
        var player: AVPlayer

        init(url: URL) {
            self.url = url
            self.player = AVPlayer(url: url)
        }
    }
}

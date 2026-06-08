import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            self.applyLaunchWindowGeometry()
        }
    }

    private func applyLaunchWindowGeometry() {
        guard let window = NSApp.windows.first(where: { $0.title == "ChatParser" }) ?? NSApp.windows.first else {
            return
        }
        let visibleFrame = window.screen?.visibleFrame ?? NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let width = min(1280, max(1040, visibleFrame.width - 80))
        let height = min(820, max(700, visibleFrame.height - 80))
        let origin = NSPoint(
            x: visibleFrame.midX - width / 2,
            y: visibleFrame.midY - height / 2
        )
        window.minSize = NSSize(width: 1040, height: 700)
        window.setFrame(NSRect(origin: origin, size: NSSize(width: width, height: height)), display: true)
    }
}

@main
struct ChatParserMacApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var state = AppState()

    var body: some Scene {
        WindowGroup("ChatParser", id: "main") {
            ContentView()
                .environmentObject(state)
                .frame(minWidth: 1040, idealWidth: 1280, minHeight: 700, idealHeight: 820)
        }
        .defaultSize(width: 1280, height: 820)
        .commands {
            CommandGroup(after: .newItem) {
                Button("Choose Export Folder...") {
                    state.chooseInputDirectory()
                }
                .keyboardShortcut("o", modifiers: [.command])

                Button(state.isRunning ? "Cancel Transformation" : "Run Transformation") {
                    if state.isRunning {
                        state.cancel()
                    } else {
                        state.run()
                    }
                }
                .keyboardShortcut("r", modifiers: [.command])
                .disabled(!state.canRun && !state.isRunning)
            }
        }

        Settings {
            SettingsView()
                .environmentObject(state)
        }
    }
}

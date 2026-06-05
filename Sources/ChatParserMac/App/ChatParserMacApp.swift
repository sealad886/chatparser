import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
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
                .frame(minWidth: 980, minHeight: 640)
        }
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

import SwiftUI

@main
struct NeuroVisWebViewApp: App {
    var body: some Scene {
        WindowGroup {
            WebPreviewView()
                .ignoresSafeArea()
                .statusBarHidden(true)
        }
    }
}

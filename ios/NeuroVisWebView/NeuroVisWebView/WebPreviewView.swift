import SwiftUI
import WebKit

struct WebPreviewView: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> FullscreenWebViewController {
        FullscreenWebViewController()
    }

    func updateUIViewController(_ viewController: FullscreenWebViewController, context: Context) {}
}

final class FullscreenWebViewController: UIViewController {
    private let previewURL = URL(string: "http://127.0.0.1:8001/?app=1")!
    private var webView: WKWebView!

    override var prefersStatusBarHidden: Bool { true }
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask { .landscape }
    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation { .landscapeRight }

    override func loadView() {
        let configuration = WKWebViewConfiguration()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = true

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.isOpaque = true
        webView.backgroundColor = UIColor(red: 0.992, green: 0.992, blue: 0.984, alpha: 1)
        webView.scrollView.backgroundColor = webView.backgroundColor
        webView.scrollView.bounces = false
        webView.scrollView.contentInsetAdjustmentBehavior = .never

        self.webView = webView
        view = webView
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = webView.backgroundColor
        webView.load(URLRequest(url: previewURL, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData))
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        webView.frame = view.bounds
    }
}

extension FullscreenWebViewController: WKNavigationDelegate {}

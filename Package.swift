// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "ChatParserMac",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "ChatParserMac", targets: ["ChatParserMac"])
    ],
    targets: [
        .executableTarget(
            name: "ChatParserMac",
            path: "Sources/ChatParserMac"
        )
    ]
)

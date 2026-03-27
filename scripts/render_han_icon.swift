import AppKit
import CoreGraphics

let sourcePath = "/Users/wangzf/Pictures/han.jpg"
let outputDir = FileManager.default.currentDirectoryPath + "/static/icons"

guard let sourceImage = NSImage(contentsOfFile: sourcePath) else {
    fputs("Failed to load source image at \(sourcePath)\n", stderr)
    exit(1)
}

let srcSize = sourceImage.size
let cropSide = min(srcSize.width, srcSize.height) * 0.55
let cropRect = NSRect(
    x: (srcSize.width - cropSide) / 2.0,
    y: (srcSize.height - cropSide) / 2.0,
    width: cropSide,
    height: cropSide
)

guard let cgImage = sourceImage.cgImage(forProposedRect: nil, context: nil, hints: nil),
      let cropped = cgImage.cropping(to: cropRect) else {
    fputs("Failed to crop source image.\n", stderr)
    exit(1)
}

func smoothstep(edge0: CGFloat, edge1: CGFloat, x: CGFloat) -> CGFloat {
    let t = max(0, min(1, (x - edge0) / (edge1 - edge0)))
    return t * t * (3 - 2 * t)
}

func makeTintMask(
    from image: CGImage,
    red: UInt8,
    green: UInt8,
    blue: UInt8,
    alphaScale: CGFloat = 1.0,
    edge0: CGFloat = 0.04,
    edge1: CGFloat = 0.22
) -> CGImage? {
    let width = image.width
    let height = image.height
    let rep = NSBitmapImageRep(cgImage: image)
    guard let source = rep.bitmapData else {
        return nil
    }

    var output = [UInt8](repeating: 0, count: width * height * 4)
    let bytesPerPixel = rep.bitsPerPixel / 8
    let bytesPerRow = rep.bytesPerRow

    for y in 0..<height {
        for x in 0..<width {
            let srcIndex = y * bytesPerRow + x * bytesPerPixel
            let dstIndex = (y * width + x) * 4

            let r = CGFloat(source[srcIndex]) / 255.0
            let g = CGFloat(source[srcIndex + 1]) / 255.0
            let b = CGFloat(source[srcIndex + 2]) / 255.0
            let luminance = 0.299 * r + 0.587 * g + 0.114 * b
            let alpha = 1.0 - smoothstep(edge0: edge0, edge1: edge1, x: luminance)
            let a = UInt8(max(0, min(255, alpha * alphaScale * 255.0)))

            output[dstIndex] = red
            output[dstIndex + 1] = green
            output[dstIndex + 2] = blue
            output[dstIndex + 3] = a
        }
    }

    guard let provider = CGDataProvider(data: Data(output) as CFData) else { return nil }
    return CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
        provider: provider,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent
    )
}

let emblemMask = makeTintMask(from: cropped, red: 255, green: 253, blue: 250, alphaScale: 1.0)
let emblemShadowMask = makeTintMask(from: cropped, red: 112, green: 40, blue: 44, alphaScale: 0.34)
let emblemDeepShadowMask = makeTintMask(from: cropped, red: 96, green: 30, blue: 34, alphaScale: 0.16)
let emblemHighlightMask = makeTintMask(from: cropped, red: 255, green: 255, blue: 255, alphaScale: 0.28)
let emblemTopHighlightMask = makeTintMask(from: cropped, red: 255, green: 255, blue: 255, alphaScale: 0.12)

func sampledBannerRed(from image: CGImage) -> NSColor {
    let rep = NSBitmapImageRep(cgImage: image)
    guard let source = rep.bitmapData else {
        return NSColor(calibratedRed: 166/255, green: 54/255, blue: 58/255, alpha: 1)
    }

    let width = image.width
    let height = image.height
    let bytesPerPixel = rep.bitsPerPixel / 8
    let bytesPerRow = rep.bytesPerRow
    var rs: CGFloat = 0
    var gs: CGFloat = 0
    var bs: CGFloat = 0
    var count: CGFloat = 0

    for y in stride(from: height / 3, to: height * 2 / 3, by: 5) {
        for x in stride(from: width / 10, to: width * 9 / 10, by: 5) {
            let idx = y * bytesPerRow + x * bytesPerPixel
            let r = CGFloat(source[idx]) / 255.0
            let g = CGFloat(source[idx + 1]) / 255.0
            let b = CGFloat(source[idx + 2]) / 255.0
            let luminance = 0.299 * r + 0.587 * g + 0.114 * b
            if luminance > 0.18 && luminance < 0.82 {
                rs += r
                gs += g
                bs += b
                count += 1
            }
        }
    }

    guard count > 0 else {
        return NSColor(calibratedRed: 166/255, green: 54/255, blue: 58/255, alpha: 1)
    }

    return NSColor(calibratedRed: rs / count, green: gs / count, blue: bs / count, alpha: 1)
}

let backgroundRed = sampledBannerRed(from: cgImage)

struct Palette {
    static let outerRing = NSColor(calibratedWhite: 1.0, alpha: 0.0)
}

func render(size: CGFloat) -> NSImage {
    let image = NSImage(size: NSSize(width: size, height: size))
    image.lockFocus()

    let rect = NSRect(x: 0, y: 0, width: size, height: size)
    NSColor.clear.setFill()
    rect.fill()

    let panelInset = size * 0.03
    let panelRect = rect.insetBy(dx: panelInset, dy: panelInset)
    let panelRadius = size * 0.22
    let panelPath = NSBezierPath(roundedRect: panelRect, xRadius: panelRadius, yRadius: panelRadius)
    backgroundRed.setFill()
    panelPath.fill()

    let emblemDiameter = size * 0.92
    let emblemRect = NSRect(
        x: (size - emblemDiameter) / 2.0,
        y: (size - emblemDiameter) / 2.0,
        width: emblemDiameter,
        height: emblemDiameter
    )

    let ringPath = NSBezierPath(ovalIn: emblemRect)

    NSGraphicsContext.current?.saveGraphicsState()
    ringPath.addClip()
    let context = NSGraphicsContext.current?.cgContext
    context?.interpolationQuality = .high
    if let emblemDeepShadowMask {
        context?.draw(emblemDeepShadowMask, in: emblemRect.offsetBy(dx: size * 0.014, dy: -size * 0.014))
    }
    if let emblemShadowMask {
        context?.draw(emblemShadowMask, in: emblemRect.offsetBy(dx: size * 0.010, dy: -size * 0.010))
    }
    if let emblemTopHighlightMask {
        context?.draw(emblemTopHighlightMask, in: emblemRect.offsetBy(dx: -size * 0.010, dy: size * 0.010))
    }
    if let emblemHighlightMask {
        context?.draw(emblemHighlightMask, in: emblemRect.offsetBy(dx: -size * 0.006, dy: size * 0.006))
    }
    if let emblemMask {
        context?.draw(emblemMask, in: emblemRect)
    }
    NSGraphicsContext.current?.restoreGraphicsState()

    image.unlockFocus()
    return image
}

func writePNG(_ image: NSImage, to path: String) throws {
    guard let tiff = image.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff),
          let data = rep.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "render_han_icon", code: 1)
    }
    try data.write(to: URL(fileURLWithPath: path))
}

let targets: [(CGFloat, String)] = [
    (16, "favicon-16x16.png"),
    (32, "favicon-32x32.png"),
    (150, "mstile-150x150.png"),
    (180, "apple-touch-icon.png"),
    (192, "android-chrome-192x192.png"),
    (512, "android-chrome-512x512.png")
]

for (size, name) in targets {
    try writePNG(render(size: size), to: outputDir + "/" + name)
}

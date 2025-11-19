# Setting Up GitHub Pages for Your Project

Follow these steps to enable the landing page:

## Step 1: Push the `docs` folder to GitHub

```bash
git add docs/
git commit -m "Add GitHub Pages landing page"
git push origin main
```

## Step 2: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** (top right)
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select:
   - **Branch**: `main`
   - **Folder**: `/docs`
5. Click **Save**

## Step 3: Wait for Deployment

- GitHub will take 1-2 minutes to build and deploy your site
- Once ready, your site will be available at:
  ```
  https://[your-username].github.io/badrobots-feat-neckface/
  ```

## Step 4: Update Links (Optional)

If you want to update the "Code" button link in the landing page:

1. Edit `docs/index.html`
2. Find this line (around line 87):
   ```html
   <a href="https://github.com/yourusername/badrobots-feat-neckface" class="btn btn-secondary" target="_blank">💻 Code</a>
   ```
3. Replace `yourusername` with your GitHub username
4. Commit and push the changes

## Customization Options

### Update Paper Links
If your PDFs are hosted elsewhere, update these lines in `docs/index.html`:
```html
<a href="../submission/main.pdf" class="btn" target="_blank">📄 Paper</a>
<a href="../submission/supp_material.pdf" class="btn btn-secondary" target="_blank">📊 Supplementary</a>
```

### Add a Video
If you have a video demo, uncomment and update the video section in `docs/index.html`:
```html
<div class="video-container">
    <iframe src="https://www.youtube.com/embed/YOUR_VIDEO_ID" frameborder="0" allowfullscreen></iframe>
</div>
```

### Change Colors
The purple gradient theme uses these colors:
- Primary: `#667eea`
- Secondary: `#764ba2`

You can change them by searching and replacing these hex codes in the `<style>` section.

## Troubleshooting

### Site not loading?
- Make sure you pushed the `docs` folder to GitHub
- Check that you selected `/docs` as the source folder, not root
- Wait a few minutes for GitHub to build the site

### Images not showing?
- The landing page uses the same GitHub-hosted image as your README
- If you want to use local images, place them in `docs/images/` and update the paths

### Want to use your own domain?
- In GitHub Pages settings, add your custom domain
- Update your DNS records (see GitHub's documentation)

## Preview Locally

To preview the site locally before pushing:

```bash
cd docs
python3 -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

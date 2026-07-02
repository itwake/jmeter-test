pipeline {
  agent any

  stages {
    stage('Trigger remote A and stream logs') {
      steps {
        script {
          def remoteCredId = 'remote-jenkins-api-token'

          withCredentials([usernamePassword(
            credentialsId: remoteCredId,
            usernameVariable: 'REMOTE_JENKINS_USER',
            passwordVariable: 'REMOTE_JENKINS_TOKEN'
          )]) {
            def handle = triggerRemoteJob(
              remoteJenkinsName: 'remoteJenkins',
              job: 'A',
              parameters: [
                ENV: 'dev'
              ],
              auth: CredentialsAuth(credentials: remoteCredId),

              // 关键：不要让 triggerRemoteJob 自己等 A 完成
              blockBuildUntilComplete: false,

              // 这里不要依赖 enhancedLogging，因为 non-blocking 模式下不适合直接打完整日志
              enhancedLogging: false,

              // 避免 remote A 失败时 trigger step 本身提前把 B 弄失败
              shouldNotFailBuild: true
            )

            echo "Remote A triggered. Initial status: ${handle.getBuildStatus()}"

            def aBuildUrl = null

            timeout(time: 30, unit: 'MINUTES') {
              waitUntil {
                handle.updateBuildStatus()

                echo "Remote A status: ${handle.getBuildStatus()}"

                try {
                  aBuildUrl = handle.getBuildUrl()?.toString()
                } catch (ignored) {
                  aBuildUrl = null
                }

                if (aBuildUrl) {
                  return true
                }

                sleep 2
                return false
              }
            }

            echo "Remote A build URL: ${aBuildUrl}"

            withEnv(["A_BUILD_URL=${aBuildUrl}"]) {
              sh '''#!/usr/bin/env bash
set -euo pipefail
set +x

offset=0

echo "========== Remote Jenkins A console log start =========="

while true; do
  headers="$(mktemp)"
  body="$(mktemp)"

  curl -sS --fail \
    -u "$REMOTE_JENKINS_USER:$REMOTE_JENKINS_TOKEN" \
    -D "$headers" \
    -o "$body" \
    "${A_BUILD_URL%/}/logText/progressiveText?start=${offset}"

  cat "$body"

  new_offset="$(awk 'tolower($1)=="x-text-size:" {gsub(/\\r/,"",$2); print $2}' "$headers" | tail -n 1)"
  more_data="$(awk 'tolower($1)=="x-more-data:" {gsub(/\\r/,"",$2); print $2}' "$headers" | tail -n 1)"

  rm -f "$headers" "$body"

  if [ -z "$new_offset" ]; then
    echo "ERROR: Jenkins did not return X-Text-Size."
    exit 1
  fi

  offset="$new_offset"

  if [ "$more_data" = "true" ]; then
    sleep 2
  else
    break
  fi
done

echo
echo "========== Remote Jenkins A console log end =========="

A_RESULT="$(curl -sS --fail \
  -u "$REMOTE_JENKINS_USER:$REMOTE_JENKINS_TOKEN" \
  "${A_BUILD_URL%/}/api/xml?xpath=string(//result)")"

echo "Remote A result: ${A_RESULT}"

case "$A_RESULT" in
  SUCCESS)
    exit 0
    ;;
  UNSTABLE)
    exit 2
    ;;
  FAILURE|ABORTED|NOT_BUILT)
    exit 1
    ;;
  *)
    echo "Unknown Remote A result: ${A_RESULT}"
    exit 1
    ;;
esac
'''
            }
          }
        }
      }
    }
  }
}
